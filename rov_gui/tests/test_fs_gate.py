#!/usr/bin/env python3
"""test_fs_gate.py — the GPU schedule gate between FoundationStereo and the
diffusion policy (rov_gui/perception/fs_gate.py) on its own: pure threading +
stdlib, no Qt, no torch, no GPU, no camera, no pytest dependency (plain-assert
functions; they collect under pytest as well).

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_fs_gate.py
    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python -m pytest \
        rov_gui/tests/test_fs_gate.py -q -p no:cacheprovider

The specification is fs_gate.py's module docstring and its method
docstrings. What is pinned here:
  * construction: `free` builds NO gate, so `free` and any name that is not a
    schedule are refused; `yield` and `only` are accepted; the default windows
    are the documented ones (wait_s 0.15, and wait_idle() with no argument is
    bounded by it); the module imports the standard library only;
  * `yield`: nothing holds -> FS_GO; hold() -> FS_HOLD until release(); a hold
    is a LEASE (older than lease_s it is ignored, counted once per hold, and a
    new hold is a new lease); the in-flight mark follows fs_begin / fs_end and
    fs_end(ran=False) clears it without counting a frame;
  * wait_idle: at once when nothing is in flight; bounded by the timeout given
    (and by wait_s when none is) while a frame stays in flight, and after such
    a timeout the hold still keeps the NEXT frame out; woken by fs_end (a
    frame's, or an empty poll's) from the other thread, not by its timeout; a
    stale in-flight mark is not waited on, the next fs_begin resets and counts
    it and that reset lets a waiting policy go; its bound and the
    milliseconds it reports are REAL time even when a clock was injected (the
    injected clock only judges staleness), so under a clock that never moves
    a wait on a live frame returns at its bound and says timeout;
  * the stereo thread never blocks: fs_begin / fs_end / snapshot return at
    once while a policy sits in wait_idle or holds;
  * atomicity under real threads: no frame STARTS inside a hold and, once
    wait_idle has returned without a timeout, the forward overlaps no frame —
    while the same two loops with no gate (the `free` schedule) do overlap, so
    the check is not vacuous; a refusal only ever happens during a hold;
  * `only`: no mission -> free; an active mission with no request -> FS_DRAIN;
    request_burst(n) -> FS_GO until n frames have been DELIVERED
    (fs_delivered), then FS_DRAIN. A burst is charged at fs_delivered, never
    at fs_end: fs_end alone spends nothing, a failed frame
    (fs_end(ran=False)) is neither a frame nor a delivery and the next pair
    runs at once, a computed frame the tap dropped (fs_end, no fs_delivered)
    spends nothing until two are delivered or burst_timeout_s voids the
    allowance; a frame in flight at the request — or one that ended before it
    but is delivered after it — is the first of the n, one delivered before
    the request is no burst's; fs_delivered outside a mission, under `yield`
    or with nothing owed only counts `delivered`. An empty poll (no pair)
    uses nothing up; a second request while one is pending is refused; an
    allowance nobody used expires after burst_timeout_s (counted once) and
    does not refuse the next request; the heartbeat, the starve watchdog (and
    a request closing it again), set_active(False) clearing the allowance,
    hold() -> FS_DRAIN in a mission and FS_HOLD outside one, the lease inside
    a mission; under real threads exactly the frames asked for are delivered
    (failed and dropped frames mixed in cost one more start each, nothing
    else), none runs during the forward;
  * `yield` ignores set_active / request_burst (False, state unchanged);
  * snapshot(): json.dumps with no `default`, counters that tell what
    happened (`frames` = maps that came out, `delivered` = those that reached
    the policy), None summaries when nothing waited or held, summaries that
    order what was recorded, read-only, bounded records; calls made out of
    order (a release with no hold, an end with no begin, a delivery with no
    frame) do not raise;
  * properties: random calls from two threads never raise and fs_begin stays
    non-blocking; random single-thread sequences on an injected clock follow
    the docstrings' rules (restated as `_Spec`, deliveries racing the
    policy's calls included); and whatever came before, a policy that goes
    silent — or keeps its heartbeat but stops asking — always gets
    FoundationStereo running again.

TIME. Everything about elapsed time runs on an injected clock (`_Clock`: time
passes only when a test says so) — except wait_idle, whose bound and whose
waited milliseconds are REAL time even under an injected clock (that clock
only says whether an in-flight mark is stale). Its real blocking is asserted
with real threads and a tolerance (SLACK_S) on every assertion.

CAUTION for whoever adds a test: under a frozen `_Clock`, a wait_idle on a
frame in flight that is not stale sits out its whole REAL bound (wait_s, 0.15 s
by default) and says timeout — so the waits made under `_Clock` here either
have nothing in flight or pass a short timeout, and the milliseconds a wait
reports are never exactly 0.0, only small. A gate that measured that bound on
the injected clock instead (as the first version did) never returns there:
that is why every wait_idle a test makes on its own thread goes through
`_wait()` / `_bounded()`, which fail the test — instead of hanging the file —
when a call does not come back, and why every test runs behind a time limit
(GUARD_S, the guard at the end of the file): a gate that blocks must FAIL
here, not hang.
"""

from __future__ import annotations

import contextlib
import functools
import json
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.perception.fs_gate import (  # noqa: E402
    BURST_FRAMES, FS_DRAIN, FS_GO, FS_HOLD, SCHEDULES, FsGate)

#: Tolerance of every assertion on a REAL wait: how late a thread may be
#: scheduled before a test calls the gate slow. A choice, not a measurement —
#: it only has to separate "returned when it should" from the failures looked
#: for (never returned / sat out a longer window), all of which are seconds.
SLACK_S = 0.25

#: "fs_begin returns at once", as two numbers (choices as well). Every call
#: must come back within BEGIN_MAX_S, and all but BEGIN_STRAGGLERS of them
#: within BEGIN_FAST_S: a test thread that the OS reschedules in the middle of
#: a call is not the gate blocking — a gate that blocks does so on every
#: refused poll, for as long as the hold, the wait or the window it sits behind
#: (3 ms and up in these tests).
BEGIN_FAST_S = 0.002
BEGIN_MAX_S = 0.02
BEGIN_STRAGGLERS = 2

#: The whole file takes a few seconds. A test that has not come back after
#: GUARD_S has met a gate that BLOCKS — the very defect this file exists to
#: catch — and is failed rather than left to hang the run (see the guard at
#: the end of the file).
GUARD_S = 20.0


# --------------------------------------------------------------------- utils
class _Clock:
    """The injected clock: time passes only when a test says so."""

    def __init__(self, t0: float = 1000.0):
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def step(self, dt: float) -> float:
        assert dt >= 0.0
        self.t += float(dt)
        return self.t


@contextlib.contextmanager
def _raises(exc, match=None):
    try:
        yield
    except exc as e:
        if match is not None:
            assert match in str(e), f"{match!r} not in {e!r}"
    else:
        raise AssertionError(f"{exc.__name__} was not raised")


def _n(gate) -> dict:
    return gate.snapshot()["counters"]


def _bounded(call, limit_s: float = 3.0, unstick=None):
    """Run ``call()`` on a daemon thread and give it ``limit_s`` of REAL time.

    A gate call that blocks would never return (see CAUTION in the module
    docstring); this turns that into a failed assertion — after ``unstick()``
    has had its chance to let the thread go — instead of a hung file. Returns
    ``(result, real_seconds)``; an exception raised by ``call`` is raised
    here."""
    box = {}

    def run():
        t0 = time.perf_counter()
        try:
            box["out"] = call()
        except BaseException as e:                               # noqa: BLE001
            box["exc"] = e
        box["dt"] = time.perf_counter() - t0

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(limit_s)
    if th.is_alive():
        if unstick is not None:
            unstick()
            th.join(limit_s)
        raise AssertionError(f"the call did not return within {limit_s} s "
                             f"of real time")
    if "exc" in box:
        raise box["exc"]
    return box["out"], box["dt"]


def _wait(gate, *timeout, limit_s: float = 5.0):
    """``gate.wait_idle(*timeout)`` behind `_bounded`: a wait that never came
    back fails the test, and the lost thread is let go by ending the frame it
    waits on. Returns ``(waited_ms, timed_out, real_seconds)``."""
    (ms, timed_out), real_s = _bounded(
        lambda: gate.wait_idle(*timeout), limit_s,
        unstick=lambda: gate.fs_end(ran=False))
    return ms, timed_out, real_s


def _assert_at_once(seconds, what: str) -> None:
    """Every stereo-side call in ``seconds`` returned at once (see
    BEGIN_FAST_S / BEGIN_MAX_S)."""
    assert seconds, what
    slow = sorted(dt for dt in seconds if dt >= BEGIN_FAST_S)
    assert max(seconds) < BEGIN_MAX_S, (
        f"{what}: one call took {max(seconds) * 1e3:.1f} ms")
    assert len(slow) <= BEGIN_STRAGGLERS, (
        f"{what}: {len(slow)} of {len(seconds)} calls took "
        f"{BEGIN_FAST_S * 1e3:.0f} ms or more (worst "
        f"{slow[-1] * 1e3:.1f} ms)")


def _alive(gate, clock, seconds: float, every: float = 0.1) -> None:
    """Let ``seconds`` pass on the fake clock with the policy tick's heartbeat
    alive (set_active(True) at least every ``every`` s) and nothing else."""
    k = max(1, int(round(seconds / every)))
    for _ in range(k):
        clock.step(seconds / k)
        gate.set_active(True)


def _ticks(gate, clock, n: int, *, beat: bool = False, frame_s: float = 0.08,
           tick_s: float = 0.005, ran: bool = True, deliver: bool = True) -> list:
    """``n`` stereo ticks the way FStereoWorker.tick drives the gate: ask; on
    FS_GO run a frame of ``frame_s`` and report its end — fs_end(ran), in the
    tick's `finally` (``ran=False``: no map came out, because there was no
    pair to start or the inference raised) — and then, when a map came out
    and ``deliver``, that it reached the policy mailbox: fs_delivered, the
    tick's last step, after the publish and the policy tap (``deliver=False``:
    the tap dropped it); on FS_DRAIN report the discarded pair. ``beat`` keeps
    the policy tick's heartbeat alive while the fake time passes. Returns the
    verdicts."""
    out = []
    for _ in range(n):
        verdict = gate.fs_begin()
        out.append(verdict)
        if verdict == FS_GO:
            clock.step(frame_s)
            if beat:
                gate.set_active(True)
            gate.fs_end(ran=ran)
            if ran and deliver:
                gate.fs_delivered()
        elif verdict == FS_DRAIN:
            gate.fs_drained()
        clock.step(tick_s)
        if beat:
            gate.set_active(True)
    return out


# -------------------------------------------------------------- construction
def test_free_and_unknown_modes_are_refused():
    """`free` builds no gate at all (both workers see fs_gate None), so
    asking for one is a wiring mistake — like any name that is not a
    schedule. The refusal names what it was given."""
    for bad in ("free", "", "FREE", "Yield", " only", "always", "none", None, 0):
        with _raises(ValueError, match=repr(str(bad))):
            FsGate(bad)


def test_yield_and_only_are_accepted():
    for mode in ("yield", "only"):
        gate = FsGate(mode)
        assert gate.mode == mode                     # the workers branch on it
        assert gate.snapshot()["mode"] == mode
    # Every schedule the CLI offers, except `free`, builds a gate.
    assert tuple(SCHEDULES) == ("free", "yield", "only")
    for mode in SCHEDULES:
        if mode != "free":
            FsGate(mode)
    assert len({FS_GO, FS_HOLD, FS_DRAIN}) == 3
    assert BURST_FRAMES == 2             # the newest depth frame + the one before


def test_default_windows_are_the_documented_ones():
    gate = FsGate("only")
    want = {"lease_s": 0.5, "wait_s": 0.15, "inflight_max_s": 2.5,
            "heartbeat_s": 1.0, "burst_timeout_s": 0.6, "starve_s": 1.5}
    snap = gate.snapshot()
    for key, value in want.items():
        assert getattr(gate, key) == value, key      # PolicyWorker reads these
        assert snap[key] == value, key               # ...and the run meta
    assert snap["burst_frames_per_request"] == BURST_FRAMES
    # The two relations the docstring's safety argument leans on: the bounded
    # wait ends well inside the lease (or the hold would lapse under a policy
    # still waiting), and inside the 2 s after which the controller calls the
    # policy worker silent.
    assert gate.wait_s < gate.lease_s
    assert gate.wait_s < 2.0


def test_gate_module_imports_only_the_standard_library():
    """ "Pure threading + stdlib: no Qt, no torch" — which is what lets the
    CLI import SCHEDULES before anything heavy, and this file run alone."""
    import ast

    import rov_gui.perception.fs_gate as mod

    roots = set()
    for node in ast.walk(ast.parse(Path(mod.__file__).read_text())):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, (
                f"relative import of {node.module!r}: that is the rov_gui "
                f"package, not the standard library")
            roots.add(node.module.split(".")[0])
    assert roots and roots <= set(sys.stdlib_module_names), sorted(roots)


# --------------------------------------------------------------------- yield
def test_yield_fs_begin_is_go_when_nothing_holds():
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    assert _ticks(gate, clock, 6) == [FS_GO] * 6     # back to back, as always
    n = _n(gate)
    assert n["frames"] == 6
    assert n["held_polls"] == 0 and n["inflight_reset"] == 0
    # The stereo worker reports deliveries under `yield` too; they are
    # counted, and charged to nothing.
    assert n["delivered"] == 6 and n["burst_frames"] == 0


def test_yield_hold_refuses_fs_begin_until_release():
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    gate.hold()
    for _ in range(10):                  # ten 5 ms stereo ticks, inside the lease
        assert gate.fs_begin() == FS_HOLD
        assert gate.snapshot()["in_flight"] is False     # a refusal marks nothing
        clock.step(0.005)
    assert gate.snapshot()["held"] is True
    gate.release()
    assert gate.snapshot()["held"] is False
    assert gate.fs_begin() == FS_GO
    n = _n(gate)
    assert n["held_polls"] == 10 and n["holds"] == 1
    assert n["lease_expired"] == 0 and n["drained"] == 0


def test_yield_hold_older_than_lease_is_ignored_and_counted_once():
    clock = _Clock()
    gate = FsGate("yield", clock=clock, lease_s=0.5)
    gate.hold()
    clock.step(0.49)
    assert gate.fs_begin() == FS_HOLD                # still inside the lease
    clock.step(0.02)                                 # 0.51 s: no release came
    assert gate.snapshot()["held"] is False
    assert _ticks(gate, clock, 4) == [FS_GO] * 4     # FoundationStereo is back
    assert _n(gate)["lease_expired"] == 1            # once, not once per poll
    gate.release()                                   # the late release: harmless
    assert _ticks(gate, clock, 1) == [FS_GO]
    assert _n(gate)["lease_expired"] == 1
    # The next hold is a new lease, and its expiry is its own count.
    gate.hold()
    assert gate.fs_begin() == FS_HOLD
    clock.step(0.6)
    assert _ticks(gate, clock, 2) == [FS_GO] * 2
    assert _n(gate)["lease_expired"] == 2


def test_yield_a_second_hold_renews_the_lease():
    clock = _Clock()
    gate = FsGate("yield", clock=clock, lease_s=0.5)
    gate.hold()
    clock.step(0.4)
    gate.hold()                                      # the next attempt's hold
    clock.step(0.4)                                  # 0.8 s after the first
    assert gate.fs_begin() == FS_HOLD
    clock.step(0.2)                                  # 0.6 s after the second
    assert gate.fs_begin() == FS_GO
    n = _n(gate)
    assert n["holds"] == 2 and n["lease_expired"] == 1


def test_fs_begin_marks_a_frame_in_flight_and_fs_end_clears_it():
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    assert gate.snapshot()["in_flight"] is False
    assert gate.fs_begin() == FS_GO
    snap = gate.snapshot()
    assert snap["in_flight"] is True
    assert snap["counters"]["frames"] == 0           # begun is not completed
    clock.step(0.08)
    gate.fs_end()
    snap = gate.snapshot()
    assert snap["in_flight"] is False
    assert snap["counters"]["frames"] == 1
    assert snap["counters"]["delivered"] == 0        # an end is not a delivery
    assert snap["counters"]["inflight_reset"] == 0   # a proper pair: no reset


def test_fs_end_ran_false_clears_the_mark_without_counting_a_frame():
    """FS_GO, then no pair in the mailbox: the tick reports ran=False."""
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    for _ in range(3):
        assert gate.fs_begin() == FS_GO
        gate.fs_end(ran=False)
        assert gate.snapshot()["in_flight"] is False
        clock.step(0.005)
    n = _n(gate)
    assert n["frames"] == 0
    assert n["inflight_reset"] == 0      # each empty poll WAS ended properly
    assert gate.fs_begin() == FS_GO
    gate.fs_end()
    assert _n(gate)["frames"] == 1


def test_release_is_idempotent_and_safe_without_a_hold():
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    gate.release()                                   # no hold at all
    gate.release()
    assert gate.snapshot()["hold_ms"]["n"] == 0      # nothing was held
    gate.hold()
    clock.step(0.02)
    gate.release()
    gate.release()                                   # e.g. _fs_idle after a finally
    snap = gate.snapshot()
    assert snap["held"] is False
    assert snap["hold_ms"]["n"] == 1                 # one hold, one duration
    assert abs(snap["hold_ms"]["max"] - 20.0) < 0.01
    assert gate.fs_begin() == FS_GO


# ----------------------------------------------------------------- wait_idle
def test_wait_idle_returns_at_once_when_nothing_is_in_flight():
    gate = FsGate("yield", wait_s=5.0)               # a wait would be seen
    ms, timed_out, dt = _wait(gate)
    assert timed_out is False and 0.0 <= ms < SLACK_S * 1e3
    assert dt < SLACK_S
    # The same after a frame has come and gone, with the hold on (the order
    # PolicyWorker._infer_once uses).
    assert gate.fs_begin() == FS_GO
    gate.fs_end()
    gate.hold()
    ms, timed_out, dt = _wait(gate)
    assert timed_out is False and dt < SLACK_S
    n = _n(gate)
    assert n["waits"] == 2 and n["wait_timeouts"] == 0


def test_wait_idle_times_out_bounded_while_a_frame_stays_in_flight():
    gate = FsGate("yield", wait_s=5.0)   # 0.03 below is the timeout GIVEN
    assert gate.fs_begin() == FS_GO      # ...and its fs_end does not come
    ms, timed_out, dt = _wait(gate, 0.03)
    assert timed_out is True
    assert 0.03 - 1e-3 <= dt < 0.03 + SLACK_S, dt    # waited it out, no longer
    assert 30.0 - 1.0 <= ms <= dt * 1e3 + 1.0, (ms, dt)
    snap = gate.snapshot()
    assert snap["counters"]["waits"] == 1
    assert snap["counters"]["wait_timeouts"] == 1
    assert snap["in_flight"] is True                 # the wait changed nothing
    assert snap["wait_ms"]["n"] == 1 and snap["wait_ms"]["max"] >= 29.0


def test_wait_idle_without_a_timeout_is_bounded_by_wait_s():
    gate = FsGate("yield", wait_s=0.04)
    assert gate.fs_begin() == FS_GO
    ms, timed_out, dt = _wait(gate)
    assert timed_out is True
    assert 0.04 - 1e-3 <= dt < 0.04 + SLACK_S, dt
    assert ms >= 39.0


def test_default_wait_s_is_0_15_and_bounds_wait_idle_without_an_argument():
    """150 ms: longer than one FoundationStereo frame at the --policy
    defaults and at the training store's setting (fs_gate.py, wait_s), far
    inside lease_s. It is what PolicyWorker._infer_once gets — it calls
    wait_idle() with no argument."""
    import inspect

    assert inspect.signature(FsGate).parameters["wait_s"].default == 0.15
    gate = FsGate("yield")                           # every window at its default
    assert gate.wait_s == 0.15 and gate.snapshot()["wait_s"] == 0.15
    assert gate.fs_begin() == FS_GO      # a frame whose end does not come
    gate.hold()
    ms, timed_out, dt = _wait(gate)      # no argument
    assert timed_out is True
    assert 0.15 - 1e-3 <= dt < 0.15 + SLACK_S, dt     # waited it out, no longer
    assert 150.0 - 1.0 <= ms <= dt * 1e3 + 1.0, (ms, dt)
    snap = gate.snapshot()
    assert snap["counters"]["wait_timeouts"] == 1 and snap["in_flight"] is True


def test_wait_idle_under_a_frozen_injected_clock_returns_at_the_real_time_bound():
    """The bound is REAL time even when a clock was injected; that clock only
    judges staleness. A clock that never moves and a frame in flight that is
    not stale: the first gate measured its bound on that clock and never
    returned. This one waits out the bound in real time, says timeout, and
    reports the real milliseconds it sat there (not 0.0 off the frozen
    clock)."""
    for mode in ("yield", "only"):
        clock = _Clock()
        gate = FsGate(mode, clock=clock, wait_s=5.0)  # 0.05 below is GIVEN
        assert gate.fs_begin() == FS_GO  # live: the fake time never moves
        gate.hold()
        ms, timed_out, dt = _wait(gate, 0.05, limit_s=3.0)
        assert timed_out is True, mode
        assert 0.05 - 1e-3 <= dt < 0.05 + SLACK_S, (mode, dt)
        assert 50.0 - 1.0 <= ms <= dt * 1e3 + 1.0, (mode, ms, dt)
        snap = gate.snapshot()
        assert snap["in_flight"] is True             # still live on its clock
        assert snap["counters"]["waits"] == 1
        assert snap["counters"]["wait_timeouts"] == 1
        assert snap["wait_ms"]["n"] == 1 and snap["wait_ms"]["max"] >= 49.0
        assert clock.t == 1000.0                     # nothing moved the fake time
        # With no argument the bound is wait_s — in real time as well.
        gate = FsGate(mode, clock=_Clock(), wait_s=0.04)
        assert gate.fs_begin() == FS_GO
        ms, timed_out, dt = _wait(gate, limit_s=3.0)
        assert timed_out is True, mode
        assert 0.04 - 1e-3 <= dt < 0.04 + SLACK_S, (mode, dt)
        assert 40.0 - 1.0 <= ms <= dt * 1e3 + 1.0, (mode, ms, dt)


def test_wait_idle_with_no_time_left_returns_at_once_and_says_timeout():
    gate = FsGate("yield", wait_s=5.0)
    assert gate.fs_begin() == FS_GO
    for timeout in (0.0, -1.0):
        ms, timed_out, dt = _wait(gate, timeout)
        assert timed_out is True and ms < SLACK_S * 1e3
        assert dt < SLACK_S
    assert _n(gate)["wait_timeouts"] == 2
    assert gate.snapshot()["in_flight"] is True


def test_after_a_wait_timeout_the_hold_still_keeps_the_next_frame_out():
    """A timeout is not a failure: the policy goes ahead beside the frame it
    could not wait out (today's behaviour) — and when that frame ends in the
    middle of the forward, no NEW one starts until the release."""
    gate = FsGate("yield", lease_s=30.0)
    assert gate.fs_begin() == FS_GO      # a long frame (the graph capture)
    gate.hold()
    ms, timed_out, _ = _wait(gate, 0.01)
    assert timed_out is True
    assert gate.snapshot()["held"] is True           # the hold outlives the wait
    gate.fs_end()                        # ...the frame ends during the forward
    assert gate.fs_begin() == FS_HOLD
    gate.release()
    assert gate.fs_begin() == FS_GO


def test_wait_idle_is_woken_promptly_by_fs_end_from_another_thread():
    # ran=False as well: a poll that found no pair is "in flight" for a
    # moment too, and its end must let a waiting policy go like a frame's.
    for ran in (True, False):
        gate = FsGate("yield", lease_s=30.0)
        assert gate.fs_begin() == FS_GO  # a frame is in flight...
        gate.hold()                      # ...when the policy's attempt begins
        marks = {}

        def stereo():
            time.sleep(0.03)             # the rest of the frame
            marks["end_called"] = time.perf_counter()
            gate.fs_end(ran=ran)

        def policy():
            t_in = time.perf_counter()
            out = gate.wait_idle(2.0)
            return out, t_in, time.perf_counter()

        th = threading.Thread(target=stereo, daemon=True)
        th.start()
        ((ms, timed_out), t_in, t_out), _ = _bounded(
            policy, 5.0, unstick=lambda: gate.fs_end(ran=False))
        th.join(5.0)
        assert not th.is_alive()
        assert timed_out is False
        t_end = marks["end_called"]
        assert t_out >= t_end                        # not before the frame ended
        assert t_out - t_end < SLACK_S               # by fs_end, not the 2 s
        # What it reports is the time it sat there: from its call to the end
        # of the frame (about 30 ms here), no less and no more.
        assert (t_end - t_in) * 1e3 - 5.0 <= ms <= (t_out - t_in) * 1e3 + 1.0, (
            ms, t_end - t_in, t_out - t_in)
        # The frame is over and the hold is still on: nothing may start now.
        assert gate.fs_begin() == FS_HOLD
        gate.release()
        assert gate.fs_begin() == FS_GO
        n = _n(gate)
        assert n["waits"] == 1 and n["wait_timeouts"] == 0
        assert n["frames"] == (1 if ran else 0)


def test_wait_idle_does_not_wait_on_a_stale_in_flight_mark():
    clock = _Clock()
    gate = FsGate("yield", clock=clock, inflight_max_s=2.5)
    assert gate.fs_begin() == FS_GO      # the tick raised: no fs_end will come
    clock.step(2.4)
    assert gate.snapshot()["in_flight"] is True      # not stale yet
    clock.step(0.2)                                  # 2.6 s old
    assert gate.snapshot()["in_flight"] is False
    gate.hold()
    # Stale on the INJECTED clock (2.6 s), while hardly any real time has
    # passed: a gate that judged staleness on real time would sit out the
    # 5 s here. What it reports is the real time it took — next to nothing.
    ms, timed_out, real_s = _wait(gate, 5.0)
    assert timed_out is False
    assert real_s < SLACK_S
    assert 0.0 <= ms <= real_s * 1e3 + 1.0, (ms, real_s)
    n = _n(gate)
    assert n["waits"] == 1 and n["wait_timeouts"] == 0
    assert n["inflight_reset"] == 0      # looking and waiting reset nothing


def test_fs_begin_resets_a_stale_in_flight_mark_and_counts_it():
    clock = _Clock()
    gate = FsGate("yield", clock=clock, inflight_max_s=2.5)
    assert gate.fs_begin() == FS_GO      # the tick raised before its fs_end
    clock.step(3.0)                      # older than inflight_max_s
    assert gate.fs_begin() == FS_GO      # the next tick is not refused...
    snap = gate.snapshot()
    assert snap["counters"]["inflight_reset"] == 1
    assert snap["in_flight"] is True     # ...and ITS frame is the one in flight
    assert snap["counters"]["frames"] == 0           # the lost frame never ended
    gate.fs_end()
    snap = gate.snapshot()
    assert snap["in_flight"] is False and snap["counters"]["frames"] == 1
    # A proper begin/end pair resets nothing.
    assert _ticks(gate, clock, 3) == [FS_GO] * 3
    assert _n(gate)["inflight_reset"] == 1
    # One thread calls fs_begin, so a mark it finds is its own lost frame
    # whatever its age: the very next tick (5 ms) takes it back too, it does
    # not sit out inflight_max_s.
    assert gate.fs_begin() == FS_GO
    clock.step(0.005)
    assert gate.fs_begin() == FS_GO
    assert _n(gate)["inflight_reset"] == 2


def test_fs_begin_reset_of_a_lost_frame_releases_a_waiting_policy():
    """The stereo tick raised between fs_begin and fs_end while the policy
    waits for that frame: the next tick's fs_begin takes the mark back, is
    itself refused by the hold, and the policy goes on at once — not after
    its timeout."""
    gate = FsGate("yield", lease_s=30.0, inflight_max_s=30.0)
    assert gate.fs_begin() == FS_GO      # ...and no fs_end
    gate.hold()
    box = {}
    th = threading.Thread(
        target=lambda: box.setdefault("out", gate.wait_idle(2.0)), daemon=True)
    th.start()
    time.sleep(0.03)                     # the policy is inside wait_idle
    assert th.is_alive()
    t0 = time.perf_counter()
    assert gate.fs_begin() == FS_HOLD    # the next stereo tick
    th.join(5.0)
    woke_s = time.perf_counter() - t0
    assert not th.is_alive()
    ms, timed_out = box["out"]
    assert timed_out is False
    assert woke_s < SLACK_S and ms < (0.03 + SLACK_S) * 1e3
    snap = gate.snapshot()
    assert snap["counters"]["inflight_reset"] == 1
    assert snap["in_flight"] is False


# ------------------------------------------------- the stereo thread's side
def test_stereo_side_never_blocks_behind_a_waiting_or_holding_policy():
    """fs_begin must keep returning to the event loop (a queued shutdown is
    delivered only between ticks), and snapshot is called from the
    controller's thread: none of them may sit behind the policy's wait."""
    gate = FsGate("yield", lease_s=30.0, inflight_max_s=30.0)
    assert gate.fs_begin() == FS_GO      # a frame in flight
    gate.hold()
    box = {}
    th = threading.Thread(
        target=lambda: box.setdefault("out", gate.wait_idle(1.0)), daemon=True)
    th.start()
    time.sleep(0.05)                     # the policy is inside wait_idle
    assert th.is_alive()
    t0 = time.perf_counter()
    snap = gate.snapshot()               # the controller's thread, for the meta
    dt_snap = time.perf_counter() - t0
    assert snap["in_flight"] is True and snap["held"] is True
    assert th.is_alive()                 # looking did not let the policy go
    t0 = time.perf_counter()
    gate.fs_end()                        # the frame ends
    dt_end = time.perf_counter() - t0
    th.join(5.0)
    assert not th.is_alive() and box["out"][1] is False
    assert dt_snap < BEGIN_MAX_S and dt_end < BEGIN_MAX_S, (dt_snap, dt_end)
    # The forward is running now (hold on, nothing in flight): the stereo
    # tick polls every 5 ms and is refused at once, every time.
    took = []
    for _ in range(200):
        t0 = time.perf_counter()
        verdict = gate.fs_begin()
        took.append(time.perf_counter() - t0)
        assert verdict == FS_HOLD
    _assert_at_once(took, "fs_begin under a hold")
    gate.release()
    assert gate.fs_begin() == FS_GO
    # `only`, a mission with nothing asked for: refused at once as well.
    only = FsGate("only")
    only.set_active(True)
    took = []
    for _ in range(200):
        t0 = time.perf_counter()
        verdict = only.fs_begin()
        took.append(time.perf_counter() - t0)
        assert verdict == FS_DRAIN
    _assert_at_once(took, "fs_begin in a mission with nothing asked for")


# ------------------------------------------------- atomicity (real threads)
def _run_yield_pair(gate, seconds: float):
    """One stereo thread (ask / "the frame" / end / delivered — the stereo
    worker reports deliveries under `yield` too) and one policy thread
    (hold / wait_idle / "the forward" / release) for ``seconds``. ``gate``
    None is the `free` schedule: the same two loops with no call into a gate.

    The stamps are taken so that a recorded interval lies INSIDE the true
    one (after the call that opens it returned, before the call that closes
    it is made): two intervals the gate keeps apart can then never be
    recorded as overlapping.

    Returns (frames, refusals, cycles, errors):
      frames    (t_ask, t_go, t_done)   fs_begin called / returned FS_GO /
                                        about to call fs_end
      refusals  (t_ask, t_back)         an fs_begin that returned FS_HOLD
      cycles    (t_pre, t_held, t_idle, t_fwd_end, t_post, timed_out)
                                        before hold() / after it / wait_idle
                                        returned / about to release() / after
    """
    frames, refusals, cycles, errors = [], [], [], []
    stop = threading.Event()

    def stereo():
        try:
            while not stop.is_set():
                t_ask = time.perf_counter()
                verdict = FS_GO if gate is None else gate.fs_begin()
                t_go = time.perf_counter()
                if verdict == FS_GO:
                    time.sleep(0.002)                # "the frame"
                    t_done = time.perf_counter()
                    if gate is not None:
                        gate.fs_end()
                        gate.fs_delivered()          # ...into the policy mailbox
                    frames.append((t_ask, t_go, t_done))
                elif verdict == FS_HOLD:
                    refusals.append((t_ask, t_go))
                    time.sleep(0.0003)               # the next poll
                else:
                    errors.append(f"stereo: fs_begin gave {verdict!r} "
                                  f"under yield")
                    return
        except BaseException as e:                               # noqa: BLE001
            errors.append(f"stereo: {type(e).__name__}: {e}")

    def policy():
        try:
            while not stop.is_set():
                t_pre = time.perf_counter()
                if gate is not None:
                    gate.hold()
                t_held = time.perf_counter()
                timed_out = False
                if gate is not None:
                    _, timed_out = gate.wait_idle(2.0)
                t_idle = time.perf_counter()
                time.sleep(0.002)                    # "the forward"
                t_fwd_end = time.perf_counter()
                if gate is not None:
                    gate.release()
                t_post = time.perf_counter()
                cycles.append((t_pre, t_held, t_idle, t_fwd_end, t_post,
                               timed_out))
                time.sleep(0.004)                    # the rest of the period
        except BaseException as e:                               # noqa: BLE001
            errors.append(f"policy: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=stereo, daemon=True),
               threading.Thread(target=policy, daemon=True)]
    for th in threads:
        th.start()
    try:
        time.sleep(seconds)
    finally:
        stop.set()
        for th in threads:
            th.join(5.0)
    if any(th.is_alive() for th in threads):
        errors.append("a thread did not stop")
    return frames, refusals, cycles, errors


def _forwards_overlapping_a_frame(frames, cycles) -> int:
    """How many forwards [t_idle, t_fwd_end] share time with a frame's
    [t_go, t_done] (a frame record is (t_ask, t_go, t_done, ...))."""
    hit = 0
    for (_pre, _held, t_idle, t_fwd_end, _post, _to) in cycles:
        if any(f[1] < t_fwd_end and f[2] > t_idle for f in frames):
            hit += 1
    return hit


def test_yield_no_frame_starts_inside_a_hold_and_no_forward_overlaps_a_frame():
    # The control: the same two loops with no gate DO overlap, so a zero
    # below is the gate's doing and not the measurement's.
    frames, _, cycles, errors = _run_yield_pair(None, 0.25)
    assert not errors, errors
    assert _forwards_overlapping_a_frame(frames, cycles) > 0

    # Windows far beyond the run, so a stalled thread cannot open the gate
    # legitimately (lease / stale mark) and blur what is being asserted.
    gate = FsGate("yield", lease_s=30.0, inflight_max_s=30.0)
    frames, refusals, cycles, errors = _run_yield_pair(gate, 1.0)
    assert not errors, errors
    # Both sides made progress: the gate neither starved FoundationStereo nor
    # stalled the policy.
    assert len(frames) >= 10 and len(cycles) >= 10, (len(frames), len(cycles))
    n = _n(gate)
    assert n["frames"] == len(frames) and n["holds"] == len(cycles)
    assert n["delivered"] == len(frames) and n["burst_frames"] == 0
    assert n["held_polls"] == len(refusals) > 0      # the hold did refuse ticks
    assert n["lease_expired"] == 0 and n["inflight_reset"] == 0
    # No wait hit its 2 s bound: every one was ended by the frame's fs_end...
    assert not any(c[5] for c in cycles)
    assert n["wait_timeouts"] == 0
    # ...and some DID wait, i.e. holds landed on a frame in flight and that
    # frame was allowed to finish.
    assert gate.snapshot()["wait_ms"]["max"] > 0.2

    # (A) No frame STARTED inside a hold: an fs_begin called after hold()
    # returned, and back before release() was called, never said FS_GO.
    started_inside = [
        (f, c) for c in cycles for f in frames
        if f[0] >= c[1] and f[1] <= c[3]]
    assert not started_inside, started_inside[:3]
    # (B) After wait_idle returned, the forward overlaps no frame at all: the
    # one in flight at the hold has ended, and none began after it.
    assert _forwards_overlapping_a_frame(frames, cycles) == 0
    # (C) And the other direction — FoundationStereo "runs as always": a
    # refusal only ever happens while a hold is on (every hold made is in
    # `cycles`: the policy loop finishes the cycle it is in before it stops).
    stray = [r for r in refusals
             if not any(r[1] >= c[0] and r[0] <= c[4] for c in cycles)]
    assert not stray, stray[:3]


# ---------------------------------------------------------------------- only
def test_only_runs_free_while_no_mission_is_active():
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    assert _ticks(gate, clock, 5) == [FS_GO] * 5
    gate.set_active(False)                           # said out loud: the same
    assert _ticks(gate, clock, 2) == [FS_GO] * 2
    snap = gate.snapshot()
    assert snap["mission_active"] is False
    n = snap["counters"]
    assert n["frames"] == 7 and n["held_polls"] == 0 and n["drained"] == 0
    assert n["burst_frames"] == 0 and n["activations"] == 0
    assert n["delivered"] == 7           # delivered, and charged to no burst


def test_only_active_mission_without_a_request_drains():
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    gate.set_active(True)
    # 0.2 s of 5 ms stereo ticks, the heartbeat alive: every waiting pair is
    # discarded and nothing is computed.
    assert _ticks(gate, clock, 40, beat=True) == [FS_DRAIN] * 40
    snap = gate.snapshot()
    assert snap["mission_active"] is True and snap["in_flight"] is False
    n = snap["counters"]
    assert n["frames"] == 0 and n["drained"] == 40 and n["held_polls"] == 40
    assert n["activations"] == 1         # 41 heartbeats, ONE mission
    assert n["starve_free"] == 0


def test_only_burst_of_two_gives_exactly_two_go_then_drain():
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    gate.set_active(True)
    assert gate.fs_begin() == FS_DRAIN
    assert gate.request_burst(2) is True
    assert gate.snapshot()["burst_allowance"] == 2
    # One frame at a time, reported the way FStereoWorker.tick does: fs_end
    # once the GPU is free — which spends NOTHING — then fs_delivered once
    # the depth map is in the policy mailbox, which is what a burst counts.
    for owed in (2, 1):
        assert gate.fs_begin() == FS_GO
        clock.step(0.08)
        gate.set_active(True)
        gate.fs_end()
        assert gate.snapshot()["burst_allowance"] == owed, owed
        gate.fs_delivered()
        assert gate.snapshot()["burst_allowance"] == owed - 1, owed
        clock.step(0.005)
        gate.set_active(True)
    assert _ticks(gate, clock, 4, beat=True) == [FS_DRAIN] * 4
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 0
    n = snap["counters"]
    assert n["bursts"] == 1 and n["burst_frames"] == 2 and n["frames"] == 2
    assert n["delivered"] == 2
    assert n["burst_void"] == 0 and n["starve_free"] == 0
    # With no argument a burst is BURST_FRAMES — what PolicyWorker asks for.
    assert gate.request_burst() is True
    assert gate.snapshot()["burst_allowance"] == BURST_FRAMES
    assert _ticks(gate, clock, 5, beat=True) == [FS_GO, FS_GO] + [FS_DRAIN] * 3


def test_only_burst_size_is_the_number_asked_for():
    for k in (1, 2, 3):
        clock = _Clock()
        gate = FsGate("only", clock=clock)
        gate.set_active(True)
        assert gate.request_burst(k) is True
        got = _ticks(gate, clock, k + 3, beat=True, frame_s=0.05)
        assert got == [FS_GO] * k + [FS_DRAIN] * 3, (k, got)
        n = _n(gate)
        assert n["burst_frames"] == k and n["delivered"] == k


def test_only_frame_in_flight_at_the_request_counts_as_the_first():
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    assert gate.fs_begin() == FS_GO      # no mission yet: free, a frame in flight
    clock.step(0.02)
    gate.set_active(True)                # the mission starts...
    assert gate.request_burst(2) is True             # ...and asks for its two
    clock.step(0.06)
    gate.set_active(True)
    gate.fs_end()                        # the frame that was in flight is over...
    assert gate.snapshot()["burst_allowance"] == 2   # ...which spends nothing
    gate.fs_delivered()                  # ...and it reaches the policy: 1 of 2
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 1
    assert snap["counters"]["burst_frames"] == 1
    clock.step(0.005)
    gate.set_active(True)
    # ONE more start, not two.
    assert _ticks(gate, clock, 4, beat=True) == [FS_GO] + [FS_DRAIN] * 3
    n = _n(gate)
    assert n["frames"] == 2 and n["burst_frames"] == 2 and n["bursts"] == 1
    assert n["delivered"] == 2


def test_only_empty_polls_do_not_use_up_the_burst():
    """Allowed, and no pair has arrived yet: fs_end(ran=False). The two
    frames are still owed when the pairs come."""
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    assert _ticks(gate, clock, 6, beat=True, frame_s=0.0, ran=False) == [FS_GO] * 6
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 2
    assert snap["counters"]["frames"] == 0
    assert snap["counters"]["delivered"] == 0
    assert snap["counters"]["burst_frames"] == 0
    assert _ticks(gate, clock, 4, beat=True) == [FS_GO, FS_GO, FS_DRAIN, FS_DRAIN]
    n = _n(gate)
    assert n["burst_frames"] == 2 and n["delivered"] == 2


# --------------------------------- `only`: a burst is frames DELIVERED (round 2)
def test_only_failed_frame_does_not_spend_the_burst_and_is_not_a_frame():
    """FS_GO, and the inference raised: FStereoWorker.tick's `finally` says
    fs_end(ran=False) and nothing reaches the policy mailbox. That frame is
    not a frame in the record and it does not use the burst up — the stereo
    network simply runs the next pair, at once — and the burst still closes
    after two deliveries."""
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    for k in range(3):                   # three bad frames in a row
        # The next pair is allowed AT ONCE: no fake time passes between the
        # failure below and this call (the GPU was freed by the fs_end).
        assert gate.fs_begin() == FS_GO, k
        clock.step(0.08)                 # it ran on the GPU, and raised
        gate.set_active(True)
        gate.fs_end(ran=False)
        snap = gate.snapshot()
        assert snap["in_flight"] is False, k
        assert snap["burst_allowance"] == 2, k       # nothing was spent
        n = snap["counters"]
        assert (n["frames"], n["delivered"], n["burst_frames"]) == (0, 0, 0), k
    assert _ticks(gate, clock, 5, beat=True) == [FS_GO, FS_GO] + [FS_DRAIN] * 3
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 0
    n = snap["counters"]
    assert n["frames"] == 2 and n["delivered"] == 2 and n["burst_frames"] == 2
    assert n["bursts"] == 1 and n["burst_void"] == 0
    assert n["inflight_reset"] == 0      # each failure WAS ended properly
    # A failure BETWEEN the two deliveries: one more start, and then closed.
    assert gate.request_burst(2) is True
    assert _ticks(gate, clock, 1, beat=True) == [FS_GO]              # 1 of 2
    assert _ticks(gate, clock, 1, beat=True, ran=False) == [FS_GO]   # raised
    assert gate.snapshot()["burst_allowance"] == 1
    assert _ticks(gate, clock, 3, beat=True) == [FS_GO, FS_DRAIN, FS_DRAIN]
    n = _n(gate)
    assert n["frames"] == 4 and n["delivered"] == 4 and n["burst_frames"] == 4


def test_only_frame_never_delivered_does_not_spend_the_burst():
    """A map came out (fs_end(), a frame) but never reached the policy
    mailbox — the tap dropped it (no grid, a refused grid, the mailbox not
    wanted), so fs_delivered was not called. It does not use the burst up:
    FoundationStereo keeps getting FS_GO until two frames have been
    DELIVERED — or until burst_timeout_s voids the allowance."""
    clock = _Clock()
    gate = FsGate("only", clock=clock, burst_timeout_s=0.6)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    assert _ticks(gate, clock, 3, beat=True, deliver=False) == [FS_GO] * 3
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 2              # three maps, none delivered
    n = snap["counters"]
    assert n["frames"] == 3 and n["delivered"] == 0 and n["burst_frames"] == 0
    # Two deliveries close it — not one more, not one less.
    assert _ticks(gate, clock, 4, beat=True) == [FS_GO, FS_GO, FS_DRAIN, FS_DRAIN]
    n = _n(gate)
    assert n["frames"] == 5 and n["delivered"] == 2 and n["burst_frames"] == 2
    assert n["burst_void"] == 0
    # The other way out: nothing is ever delivered. FS_GO for as long as the
    # allowance lives — every 85 ms (an 80 ms frame + a 5 ms tick), so at
    # 0, 0.085, ..., 0.595 s after the request — and FS_DRAIN from the first
    # poll past burst_timeout_s (0.68 s), the allowance void and counted once.
    assert gate.request_burst(2) is True
    got = _ticks(gate, clock, 10, beat=True, deliver=False)
    assert got == [FS_GO] * 8 + [FS_DRAIN] * 2, got
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 0
    n = snap["counters"]
    assert n["burst_void"] == 1 and n["burst_frames"] == 2 and n["bursts"] == 2
    assert n["frames"] == 13 and n["delivered"] == 2
    assert n["starve_free"] == 0         # it was the void, not the watchdog
    assert gate.request_burst(2) is True             # the policy asks again


def test_only_frame_delivered_after_the_request_is_charged_even_if_it_ended_before():
    """The stereo tick runs fs_end, then the publish and the policy tap, then
    fs_delivered; the policy tick's request can land anywhere in between. A
    frame that is in the policy mailbox only AFTER the request is the first
    of that burst for the policy worker (it counts the frames that ARRIVE
    after its request) — and for the gate: 2 -> 1, whether the frame ended
    before the request or was still in flight at it. One delivered BEFORE
    the request belongs to no burst."""
    # (1) Ended before the request, delivered after it — the frame was
    # running free when the mission started.
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    assert gate.fs_begin() == FS_GO      # no mission yet: free
    clock.step(0.08)
    gate.fs_end()                        # the map is out; publish + tap next...
    gate.set_active(True)                # ...and the policy tick starts the
    assert gate.request_burst(2) is True             # mission and asks
    assert gate.snapshot()["burst_allowance"] == 2   # the fs_end spent nothing
    gate.fs_delivered()                  # the tap puts it in the mailbox
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 1              # 1 of 2
    assert snap["counters"]["burst_frames"] == 1
    clock.step(0.005)
    gate.set_active(True)
    assert _ticks(gate, clock, 3, beat=True) == [FS_GO, FS_DRAIN, FS_DRAIN]
    n = _n(gate)
    assert n["frames"] == 2 and n["delivered"] == 2 and n["burst_frames"] == 2

    # (2) The same race inside a mission — a frame the starve watchdog let
    # run (nobody had asked for starve_s) — and, in the next burst, a frame
    # still in flight at the request. Both are the first of their burst.
    clock = _Clock()
    gate = FsGate("only", clock=clock, starve_s=1.5)
    gate.set_active(True)
    _alive(gate, clock, 1.6)             # a mission, and nobody asks
    assert gate.fs_begin() == FS_GO      # the watchdog opened the gate
    clock.step(0.08)
    gate.set_active(True)
    gate.fs_end()
    assert gate.request_burst(2) is True             # lands before the tap
    assert gate.snapshot()["burst_allowance"] == 2
    gate.fs_delivered()
    assert gate.snapshot()["burst_allowance"] == 1
    clock.step(0.005)
    gate.set_active(True)
    assert _ticks(gate, clock, 3, beat=True) == [FS_GO, FS_DRAIN, FS_DRAIN]
    _alive(gate, clock, 1.6)             # nobody asks again: watchdog again
    assert gate.fs_begin() == FS_GO      # ...this frame is in flight when
    clock.step(0.03)
    gate.set_active(True)
    assert gate.request_burst(2) is True             # ...the request comes
    clock.step(0.05)
    gate.set_active(True)
    gate.fs_end()
    assert gate.snapshot()["burst_allowance"] == 2   # an end is no delivery
    gate.fs_delivered()
    assert gate.snapshot()["burst_allowance"] == 1   # 1 of 2
    clock.step(0.005)
    gate.set_active(True)
    assert _ticks(gate, clock, 3, beat=True) == [FS_GO, FS_DRAIN, FS_DRAIN]
    n = _n(gate)
    assert n["bursts"] == 2 and n["burst_frames"] == 4 and n["delivered"] == 4
    assert n["starve_free"] == 2 and n["burst_void"] == 0

    # (3) Delivered BEFORE the request: no burst's. The request then owes
    # two frames that arrive after it, as asked.
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    assert _ticks(gate, clock, 1) == [FS_GO]         # ended AND delivered...
    gate.set_active(True)
    assert gate.request_burst(2) is True             # ...before this
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 2
    assert snap["counters"]["delivered"] == 1
    assert snap["counters"]["burst_frames"] == 0
    assert _ticks(gate, clock, 4, beat=True) == [FS_GO, FS_GO, FS_DRAIN, FS_DRAIN]
    assert _n(gate)["burst_frames"] == 2


def test_fs_delivered_outside_a_burst_only_counts_delivered():
    """fs_delivered charges a burst only under `only`, during a mission,
    with frames owed. Anywhere else — under `yield`, outside a mission (none
    yet, one whose heartbeat stopped, one that was ended), with nothing owed
    or a burst already complete — it moves the `delivered` count and NOTHING
    else: no allowance (never below zero), no burst_frames, no verdict."""

    def only_delivered_moves(gate, k: int = 3) -> None:
        before = gate.snapshot()
        for _ in range(k):
            gate.fs_delivered()
        after = gate.snapshot()
        want = json.loads(json.dumps(before))        # a deep copy
        want["counters"]["delivered"] += k
        assert after == want, (before, after)

    # `yield`: a fresh gate, and one that has run frames under a hold.
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    only_delivered_moves(gate)
    assert _ticks(gate, clock, 2) == [FS_GO] * 2
    gate.hold()
    only_delivered_moves(gate)
    gate.release()
    assert gate.fs_begin() == FS_GO

    # `only`, no mission, with an allowance granted (a request needs no
    # mission): the allowance stays.
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    assert gate.request_burst(2) is True
    only_delivered_moves(gate)
    assert gate.snapshot()["burst_allowance"] == 2

    # `only`, a mission with nothing asked for: nothing owed, nothing taken.
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    gate.set_active(True)
    only_delivered_moves(gate)
    assert gate.snapshot()["burst_allowance"] == 0
    assert gate.fs_begin() == FS_DRAIN               # the verdict is unchanged

    # A burst already complete: extra deliveries take nothing more.
    assert gate.request_burst(2) is True
    assert _ticks(gate, clock, 3, beat=True) == [FS_GO, FS_GO, FS_DRAIN]
    only_delivered_moves(gate)
    assert gate.snapshot()["burst_allowance"] == 0
    assert gate.fs_begin() == FS_DRAIN

    # A mission whose heartbeat has stopped, frames still owed (asked for
    # after the last beat, so the allowance itself is alive): not a mission.
    clock = _Clock()
    gate = FsGate("only", clock=clock, heartbeat_s=1.0, burst_timeout_s=0.6)
    gate.set_active(True)
    clock.step(0.9)
    assert gate.request_burst(2) is True
    clock.step(0.2)                      # 1.1 s since the last beat, 0.2 s
    snap = gate.snapshot()               # since the request
    assert snap["mission_active"] is False and snap["burst_allowance"] == 2
    only_delivered_moves(gate)

    # A mission that was ended: set_active(False) cleared the allowance.
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    gate.set_active(False)
    only_delivered_moves(gate)
    assert gate.snapshot()["burst_allowance"] == 0


def test_only_second_request_while_one_is_pending_is_refused():
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    assert gate.request_burst(2) is False            # nothing has run yet
    assert _ticks(gate, clock, 1, beat=True) == [FS_GO]
    assert gate.request_burst(2) is False            # one of the two still owed
    assert gate.snapshot()["burst_allowance"] == 1   # a refusal changes nothing
    assert _ticks(gate, clock, 1, beat=True) == [FS_GO]
    assert gate.request_burst(2) is True             # both ran: ask again
    n = _n(gate)
    assert n["bursts"] == 2 and n["burst_void"] == 0


def test_only_unused_allowance_expires_after_burst_timeout_and_is_counted():
    clock = _Clock()
    gate = FsGate("only", clock=clock, burst_timeout_s=0.6)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    # No pair comes from the camera: for 0.5 s every poll is allowed and
    # finds nothing.
    for _ in range(5):
        _alive(gate, clock, 0.1)
        assert gate.fs_begin() == FS_GO
        gate.fs_end(ran=False)
    assert gate.snapshot()["burst_allowance"] == 2
    _alive(gate, clock, 0.11)                        # 0.61 s after the request
    assert gate.fs_begin() == FS_DRAIN               # the allowance is gone
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 0
    assert snap["counters"]["burst_void"] == 1
    assert snap["counters"]["burst_frames"] == 0
    assert gate.fs_begin() == FS_DRAIN
    assert _n(gate)["burst_void"] == 1               # counted once
    assert gate.request_burst(2) is True             # and the policy asks again
    assert _ticks(gate, clock, 3, beat=True) == [FS_GO, FS_GO, FS_DRAIN]


def test_only_expired_allowance_does_not_refuse_the_next_request():
    """The stereo thread never polled in between (it was inside one long
    frame): the request itself sees that the old allowance is dead."""
    clock = _Clock()
    gate = FsGate("only", clock=clock, burst_timeout_s=0.6)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    _alive(gate, clock, 0.5)
    assert gate.request_burst(2) is False            # 0.5 s: still pending
    _alive(gate, clock, 0.2)                         # 0.7 s
    assert gate.request_burst(2) is True
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 2
    assert snap["counters"]["bursts"] == 2
    assert snap["counters"]["burst_void"] == 1


def test_only_heartbeat_silence_reopens_the_gate():
    clock = _Clock()
    # starve_s out of the way: what opens the gate here is the heartbeat.
    gate = FsGate("only", clock=clock, heartbeat_s=1.0, starve_s=100.0)
    gate.set_active(True)
    assert gate.fs_begin() == FS_DRAIN
    clock.step(0.9)                      # the policy tick is late, not dead
    assert gate.snapshot()["mission_active"] is True
    assert gate.fs_begin() == FS_DRAIN
    clock.step(0.2)                      # 1.1 s with no set_active(True)
    assert gate.snapshot()["mission_active"] is False
    assert _ticks(gate, clock, 3) == [FS_GO] * 3     # free again
    n = _n(gate)
    assert n["starve_free"] == 0 and n["burst_frames"] == 0 and n["frames"] == 3
    # The heartbeat coming back closes it again — as a new mission.
    gate.set_active(True)
    assert gate.fs_begin() == FS_DRAIN
    assert _n(gate)["activations"] == 2


def test_only_starve_watchdog_opens_and_a_new_request_closes_it():
    clock = _Clock()
    gate = FsGate("only", clock=clock, starve_s=1.5)
    gate.set_active(True)
    for _ in range(14):                  # 1.4 s: heartbeat alive, nobody asks
        assert gate.fs_begin() == FS_DRAIN
        _alive(gate, clock, 0.1)
    assert gate.fs_begin() == FS_DRAIN
    _alive(gate, clock, 0.2)             # 1.6 s without a request
    assert gate.snapshot()["mission_active"] is True
    assert _ticks(gate, clock, 3, beat=True) == [FS_GO] * 3
    n = _n(gate)
    assert n["starve_free"] == 1         # one episode, however many frames
    assert n["burst_frames"] == 0 and n["frames"] == 3
    # A request ends the episode: its two frames, then closed again.
    assert gate.request_burst(2) is True
    assert _ticks(gate, clock, 5, beat=True) == [FS_GO, FS_GO] + [FS_DRAIN] * 3
    assert _n(gate)["burst_frames"] == 2
    # ...and the watchdog is armed again, counted from that request (0.185 s
    # of it are gone in the five ticks above).
    _alive(gate, clock, 1.2)             # 1.385 s after the request
    assert gate.fs_begin() == FS_DRAIN
    _alive(gate, clock, 0.2)             # 1.585 s
    assert _ticks(gate, clock, 2, beat=True) == [FS_GO] * 2
    assert _n(gate)["starve_free"] == 2


def test_only_set_active_false_clears_the_allowance():
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    assert gate.snapshot()["burst_allowance"] == 2
    gate.set_active(False)               # idle / loading / a swap / teardown
    snap = gate.snapshot()
    assert snap["burst_allowance"] == 0 and snap["mission_active"] is False
    assert _ticks(gate, clock, 2) == [FS_GO] * 2     # free again...
    n = _n(gate)
    assert n["delivered"] == 2           # ...delivered, and not "burst frames"
    assert n["burst_frames"] == 0
    gate.set_active(True)                # the next mission starts owed nothing
    assert _ticks(gate, clock, 3, beat=True) == [FS_DRAIN] * 3
    assert _n(gate)["activations"] == 2


def test_only_hold_drains_during_a_mission_and_holds_outside_one():
    clock = _Clock()
    gate = FsGate("only", clock=clock)
    # Outside a mission: as under `yield`, the waiting pair is kept.
    gate.hold()
    assert gate.fs_begin() == FS_HOLD
    gate.release()
    assert _ticks(gate, clock, 1) == [FS_GO]
    # During one — even with frames owed — the pair is discarded.
    gate.set_active(True)
    assert gate.request_burst(2) is True
    gate.hold()
    assert gate.fs_begin() == FS_DRAIN
    assert gate.snapshot()["burst_allowance"] == 2   # the hold ate no frame
    gate.release()
    assert _ticks(gate, clock, 3, beat=True) == [FS_GO, FS_GO, FS_DRAIN]
    # A mission whose heartbeat has stopped is not a mission.
    clock.step(1.1)
    assert gate.snapshot()["mission_active"] is False
    gate.hold()
    assert gate.fs_begin() == FS_HOLD
    gate.release()
    assert gate.fs_begin() == FS_GO


def test_only_hold_is_a_lease_inside_a_mission_too():
    clock = _Clock()
    gate = FsGate("only", clock=clock, lease_s=0.5, burst_timeout_s=0.6)
    gate.set_active(True)
    assert gate.request_burst(2) is True
    gate.hold()                          # ...and its release never comes
    _alive(gate, clock, 0.3)
    assert gate.fs_begin() == FS_DRAIN
    _alive(gate, clock, 0.25)            # 0.55 s: past the lease, burst alive
    assert gate.fs_begin() == FS_GO
    n = _n(gate)
    assert n["lease_expired"] == 1 and n["held_polls"] == 1


def _run_only_pair(gate, seconds: float, fate):
    """PolicyWorker._tick_only's loop against FStereoWorker.tick's, on real
    threads, for ``seconds``: the policy keeps the heartbeat, asks for two,
    waits until both have ARRIVED (been delivered), then hold / wait /
    forward / release. ``fate(k)`` says what became of the k-th frame the
    gate let start: "delivered" (fs_end, then fs_delivered), "failed" (the
    inference raised: fs_end(ran=False), nothing delivered) or "dropped" (a
    map, but the tap dropped it: fs_end() and no fs_delivered).

    Returns (frames, asks, cycles, errors): frames (t_ask, t_go, t_done,
    fate) stamped as in _run_yield_pair; asks the time just before each
    granted request; cycles (t_req, t_held, t_idle, t_fwd_end, t_post,
    timed_out)."""
    frames, asks, cycles, errors = [], [], [], []
    arrived = [0]                        # frames in the policy's mailbox
    stop = threading.Event()

    def stereo():
        k = 0
        try:
            while not stop.is_set():
                t_ask = time.perf_counter()
                verdict = gate.fs_begin()
                t_go = time.perf_counter()
                if verdict == FS_GO:
                    what = fate(k)
                    k += 1
                    time.sleep(0.001)                # "the frame"
                    t_done = time.perf_counter()
                    gate.fs_end(ran=what != "failed")
                    frames.append((t_ask, t_go, t_done, what))
                    if what == "delivered":
                        gate.fs_delivered()          # ...into the mailbox
                        arrived[0] += 1
                elif verdict == FS_DRAIN:
                    gate.fs_drained()
                    time.sleep(0.0003)
                else:
                    errors.append("stereo: FS_HOLD inside an active mission")
                    return
        except BaseException as e:                               # noqa: BLE001
            errors.append(f"stereo: {type(e).__name__}: {e}")

    def policy():
        try:
            while not stop.is_set():
                gate.set_active(True)                # the heartbeat
                base = arrived[0]
                t_req = time.perf_counter()
                if not gate.request_burst(BURST_FRAMES):
                    errors.append("policy: a request was refused with "
                                  "nothing pending")
                    return
                asks.append(t_req)
                give_up = time.perf_counter() + 3.0
                while arrived[0] < base + BURST_FRAMES:  # ...until both arrived
                    if stop.is_set():
                        return
                    if time.perf_counter() > give_up:
                        errors.append("policy: the two frames never came")
                        return
                    gate.set_active(True)
                    time.sleep(0.0005)
                gate.hold()
                t_held = time.perf_counter()
                _, timed_out = gate.wait_idle(2.0)
                t_idle = time.perf_counter()
                time.sleep(0.002)                    # "the forward"
                t_fwd_end = time.perf_counter()
                gate.release()
                cycles.append((t_req, t_held, t_idle, t_fwd_end,
                               time.perf_counter(), timed_out))
                time.sleep(0.002)
        except BaseException as e:                               # noqa: BLE001
            errors.append(f"policy: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=stereo, daemon=True),
               threading.Thread(target=policy, daemon=True)]
    for th in threads:
        th.start()
    try:
        time.sleep(seconds)
    finally:
        stop.set()
        for th in threads:
            th.join(5.0)
    if any(th.is_alive() for th in threads):
        errors.append("a thread did not stop")
    return frames, asks, cycles, errors


def test_only_under_threads_runs_exactly_the_frames_asked_for_and_none_in_the_forward():
    """PolicyWorker._tick_only's loop against FStereoWorker.tick's, on real
    threads (_run_only_pair) — once with every frame delivered, once with
    failed and dropped frames mixed in: those spend nothing, so every burst
    is still exactly two DELIVERED frames, with one more start for each frame
    that did not get there."""
    patterns = (
        ("all delivered", lambda k: "delivered"),
        ("failed and dropped mixed in",
         lambda k: ("failed", "delivered", "dropped", "delivered",
                    "delivered")[k % 5]),
    )
    for name, fate in patterns:
        # Windows far beyond the run: a stalled thread must not open the
        # gate legitimately (heartbeat / starve / burst timeout / lease).
        gate = FsGate("only", lease_s=30.0, inflight_max_s=30.0,
                      heartbeat_s=30.0, burst_timeout_s=30.0, starve_s=30.0)
        gate.set_active(True)            # the mission is on before any tick
        frames, asks, cycles, errors = _run_only_pair(gate, 0.5, fate)
        assert not errors, (name, errors)
        assert len(asks) >= 5 and len(cycles) >= 5, (name, len(asks),
                                                     len(cycles))
        fates = Counter(f[3] for f in frames)
        if name != "all delivered":
            assert fates["failed"] > 0 and fates["dropped"] > 0, fates
        n = _n(gate)
        assert n["bursts"] == len(asks), name
        # `frames` counts the maps that came out, `delivered` the ones that
        # reached the policy — and a burst is charged with exactly those.
        assert n["frames"] == fates["delivered"] + fates["dropped"], (name, n)
        assert n["delivered"] == fates["delivered"] == n["burst_frames"], (
            name, n, fates)
        # ONLY what was asked for: two deliveries per request (the last
        # request may have been cut off by the stop)...
        assert 2 * (len(asks) - 1) <= n["delivered"] <= 2 * len(asks), name
        assert n["starve_free"] == 0 and n["burst_void"] == 0, name
        assert n["drained"] > 0, name    # between bursts the pairs were dropped
        # ...exactly two between one request and the next, and the burst
        # closes on its second delivery: nothing starts after it.
        for i in range(len(asks) - 1):
            inside = [f for f in frames if asks[i] < f[1] < asks[i + 1]]
            got = sum(1 for f in inside if f[3] == "delivered")
            assert got == BURST_FRAMES, (name, i, got, inside)
            assert inside[-1][3] == "delivered", (name, i, inside)
        # None started inside a hold, none overlaps a forward (a failed
        # frame held the GPU as long as any).
        assert not any(c[5] for c in cycles), name
        assert not [(f, c) for c in cycles for f in frames
                    if f[0] >= c[1] and f[1] <= c[3]], name
        assert _forwards_overlapping_a_frame(frames, cycles) == 0, name


# ------------------------------------------------- yield ignores `only`'s API
def test_yield_ignores_set_active_and_request_burst():
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    assert _ticks(gate, clock, 2) == [FS_GO] * 2
    before = gate.snapshot()
    assert gate.set_active(True) is None
    assert gate.request_burst(2) is False            # nothing to ask for
    assert gate.request_burst() is False
    assert gate.set_active(False) is None
    gate.set_active(True)
    assert gate.snapshot() == before                 # state AND counters
    assert before["mission_active"] is False and before["burst_allowance"] == 0
    # ...so FoundationStereo still runs as always, and a hold is FS_HOLD (the
    # pair is kept), never FS_DRAIN.
    assert _ticks(gate, clock, 3) == [FS_GO] * 3
    gate.hold()
    assert gate.fs_begin() == FS_HOLD
    gate.release()
    n = _n(gate)
    assert n["bursts"] == 0 and n["activations"] == 0 and n["drained"] == 0
    assert n["delivered"] == 5 and n["burst_frames"] == 0


# ------------------------------------------------------------------ snapshot
def _story(gate):
    """A scripted run of an `only` gate that touches every counter but the
    four that need time to pass (lease_expired, wait_timeouts, burst_void,
    starve_free); the clock does not move. Returns ALL the counters it must
    leave."""
    assert gate.fs_begin() == FS_GO                  # free: no mission yet
    gate.fs_end()                                    # frames 1
    gate.fs_delivered()                              # delivered 1: no burst's
    gate.set_active(True)                            # activations 1
    assert gate.fs_begin() == FS_DRAIN               # held_polls 1
    gate.fs_drained()                                # drained 1
    assert gate.request_burst(2) is True             # bursts 1
    assert gate.request_burst(2) is False
    assert gate.fs_begin() == FS_GO
    gate.fs_end()                                    # frames 2
    gate.fs_delivered()                              # delivered 2, burst_frames 1
    assert gate.fs_begin() == FS_GO
    gate.fs_end(ran=False)                           # an empty poll / a failure
    assert gate.fs_begin() == FS_GO
    gate.fs_end()                                    # frames 3 — the tap drops it
    assert gate.fs_begin() == FS_GO                  # ...the tick raises: no end
    assert gate.fs_begin() == FS_GO                  # inflight_reset 1
    gate.fs_end()                                    # frames 4
    gate.fs_delivered()                              # delivered 3, burst_frames 2
    gate.hold()                                      # holds 1
    assert gate.fs_begin() == FS_DRAIN               # held_polls 2
    ms, timed_out, _ = _wait(gate)                   # waits 1
    assert timed_out is False and 0.0 <= ms < SLACK_S * 1e3
    gate.release()
    assert gate.fs_begin() == FS_DRAIN               # held_polls 3
    gate.fs_drained(2)                               # drained 3
    return {"frames": 4, "delivered": 3, "held_polls": 3, "drained": 3,
            "holds": 1, "lease_expired": 0, "waits": 1, "wait_timeouts": 0,
            "inflight_reset": 1, "bursts": 1, "burst_frames": 2,
            "burst_void": 0, "starve_free": 0, "activations": 1}


def test_snapshot_is_json_serialisable_without_a_default():
    for mode in ("yield", "only"):
        fresh = FsGate(mode).snapshot()
        assert json.loads(json.dumps(fresh, allow_nan=False)) == fresh
    gate = FsGate("only", clock=_Clock())
    _story(gate)
    snap = gate.snapshot()
    text = json.dumps(snap, allow_nan=False)         # no `default=`, no NaN
    assert json.loads(text) == snap
    for key in ("held", "in_flight", "mission_active"):
        assert isinstance(snap[key], bool), key
    assert isinstance(snap["burst_allowance"], int)
    assert all(type(v) is int for v in snap["counters"].values())
    for key in ("mode", "counters", "wait_ms", "hold_ms",
                "burst_frames_per_request"):
        assert key in snap, key


def test_snapshot_counts_delivered_and_still_serialises_without_a_default():
    """The run meta's record of what reached the policy: `delivered`, an int
    from the first snapshot on, apart from `frames` (the maps that came
    out), and the snapshot still goes through json.dumps with no
    `default=`."""
    for mode in ("yield", "only"):
        clock = _Clock()
        gate = FsGate(mode, clock=clock)
        fresh = gate.snapshot()
        assert fresh["counters"]["delivered"] == 0, mode
        assert type(fresh["counters"]["delivered"]) is int, mode
        assert json.loads(json.dumps(fresh, allow_nan=False)) == fresh
        assert _ticks(gate, clock, 3) == [FS_GO] * 3             # delivered
        assert _ticks(gate, clock, 2, deliver=False) == [FS_GO] * 2  # dropped
        assert _ticks(gate, clock, 1, ran=False) == [FS_GO]      # failed
        snap = gate.snapshot()
        n = snap["counters"]
        assert (n["frames"], n["delivered"], n["burst_frames"]) == (5, 3, 0), (
            mode, n)
        assert type(n["delivered"]) is int, mode
        text = json.dumps(snap, allow_nan=False)     # no `default=`, no NaN
        assert json.loads(text) == snap, mode
    gate = FsGate("only", clock=_Clock())
    _story(gate)
    snap = gate.snapshot()
    assert snap["counters"]["delivered"] == 3 and snap["counters"]["frames"] == 4
    assert json.loads(json.dumps(snap, allow_nan=False)) == snap


def test_snapshot_counters_reflect_what_happened():
    gate = FsGate("only", clock=_Clock())
    assert not any(_n(gate).values())                # a new gate: all zero
    want = _story(gate)
    got = _n(gate)
    assert got == want                   # every counter, and no other
    snap = gate.snapshot()
    assert snap["mission_active"] is True and snap["held"] is False
    assert snap["in_flight"] is False and snap["burst_allowance"] == 0
    assert snap["wait_ms"]["n"] == 1 and snap["hold_ms"]["n"] == 1
    # The dict handed out is a copy: editing it does not edit the gate.
    got["frames"] = 999
    assert _n(gate)["frames"] == 4


def test_snapshot_wait_and_hold_summaries_are_none_when_nothing_waited():
    """None, not 0.0: a zero would read as "waited, and it was nothing"."""
    empty = {"p50": None, "p90": None, "max": None, "n": 0}
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    assert _ticks(gate, clock, 3) == [FS_GO] * 3     # frames alone: no waits
    snap = gate.snapshot()
    assert snap["wait_ms"] == empty and snap["hold_ms"] == empty
    gate.hold()
    ms, timed_out, _ = _wait(gate)       # nothing in flight: at once
    assert timed_out is False
    clock.step(0.025)                                # "the forward"
    gate.release()
    snap = gate.snapshot()
    # One wait: a number now, not None — the real milliseconds it took
    # (next to nothing), the same at every quantile.
    assert snap["wait_ms"]["n"] == 1
    for key in ("p50", "p90", "max"):
        assert snap["wait_ms"][key] == round(ms, 2), (key, snap["wait_ms"], ms)
    assert 0.0 <= snap["wait_ms"]["max"] < SLACK_S * 1e3
    assert snap["hold_ms"]["n"] == 1
    for key in ("p50", "p90", "max"):
        assert abs(snap["hold_ms"][key] - 25.0) < 0.01, (key, snap["hold_ms"])


def test_snapshot_summaries_order_the_recorded_milliseconds():
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    for ms in (70, 10, 100, 30, 50, 20, 90, 40, 80, 60):       # ten holds
        gate.hold()
        clock.step(ms / 1e3)
        gate.release()
    hold = gate.snapshot()["hold_ms"]
    assert hold["n"] == 10
    assert abs(hold["max"] - 100.0) < 0.01           # the longest, not the last
    assert 50.0 - 0.01 <= hold["p50"] <= 60.0 + 0.01
    assert 90.0 - 0.01 <= hold["p90"] <= 100.0 + 0.01
    assert hold["p50"] <= hold["p90"] <= hold["max"]


def test_snapshot_is_read_only():
    """Looking must not consume the once-per-hold lease count, nor reset a
    lost frame's mark: both belong to fs_begin."""
    clock = _Clock()
    gate = FsGate("yield", clock=clock, lease_s=0.5)
    assert gate.fs_begin() == FS_GO      # ...and no fs_end
    gate.hold()
    clock.step(0.6)                      # the lease has lapsed
    first = gate.snapshot()
    for _ in range(5):
        assert gate.snapshot() == first
    assert first["held"] is False and first["in_flight"] is True
    assert first["counters"]["lease_expired"] == 0
    assert first["counters"]["inflight_reset"] == 0
    assert gate.fs_begin() == FS_GO      # the stereo tick is what counts them
    n = _n(gate)
    assert n["lease_expired"] == 1 and n["inflight_reset"] == 1


def test_wait_and_hold_records_stay_bounded_over_a_long_run():
    """Two plans a second for hours: the counters keep counting, the
    millisecond records behind the summaries do not grow with them."""
    clock = _Clock()
    gate = FsGate("yield", clock=clock)
    k = 3000

    def run():
        for _ in range(k):
            gate.hold()
            gate.wait_idle()             # nothing in flight: returns at once
            clock.step(0.02)
            gate.release()

    _bounded(run, 20.0)
    snap = gate.snapshot()
    assert snap["counters"]["waits"] == k and snap["counters"]["holds"] == k
    assert 0 < snap["wait_ms"]["n"] < k
    assert 0 < snap["hold_ms"]["n"] < k
    json.dumps(snap, allow_nan=False)


def test_calls_out_of_order_do_not_raise():
    """A swallowed tick exception or a teardown can leave any call without
    its partner; none of them may raise (TimerWorker._tick would swallow it
    and the frame, or the plan, with it)."""
    for mode in ("yield", "only"):
        gate = FsGate(mode)
        gate.release()                               # no hold
        gate.fs_end()                                # no frame
        gate.fs_end(ran=False)
        gate.fs_delivered()                          # no frame, no mission
        gate.fs_drained(0)
        ms, timed_out, _ = _wait(gate, 0.0)          # nothing in flight
        assert timed_out is False and ms >= 0.0
        gate.set_active(False)                       # no mission
        gate.fs_delivered()
        gate.request_burst(0)
        gate.request_burst(-1)
        gate.hold()
        gate.hold()
        gate.release()
        gate.release()
        gate.set_active(False)
        assert gate.fs_begin() == FS_GO              # and it still works
        gate.fs_end()
        gate.fs_delivered()
        gate.fs_delivered()                          # twice for one frame
        snap = gate.snapshot()
        json.dumps(snap, allow_nan=False)
        assert snap["burst_allowance"] >= 0
        assert snap["counters"]["delivered"] == 4


# -------------------------------------------- property: two threads, real time
def _hammer(mode: str, seed: int, n_ops: int = 2500) -> dict:
    """Random gate calls from two threads — the stereo side and the policy
    side, ``n_ops`` each at least — while this thread reads snapshots like
    the controller does. The windows are a few milliseconds, so every expiry
    is crossed in real time. The stereo side reports most maps as delivered
    (fs_delivered after fs_end, as FStereoWorker.tick does). Both sides also
    make the calls a swallowed exception would leave behind (an fs_end nobody
    began, a begin with no end, a delivery with no frame, a release with no
    hold); the policy side also runs whole attempts (hold / wait / a 3 ms
    "forward" / release), so that a stereo call which blocked behind one
    would show."""
    gate = FsGate(mode, lease_s=0.004, wait_s=0.003, inflight_max_s=0.006,
                  heartbeat_s=0.003, burst_timeout_s=0.002, starve_s=0.005)
    errors, begin_s, wait_over_s = [], [], []
    verdicts = Counter()
    ops = [0, 0]
    done = (threading.Event(), threading.Event())
    give_up = time.perf_counter() + 8.0

    def stereo():
        rng = random.Random(seed)
        try:
            while time.perf_counter() < give_up:
                r = rng.random()
                if r < 0.70:
                    t0 = time.perf_counter()
                    verdict = gate.fs_begin()
                    begin_s.append(time.perf_counter() - t0)
                    verdicts[verdict] += 1
                    if verdict == FS_GO:
                        if rng.random() < 0.25:      # "the frame"
                            time.sleep(rng.choice((0.0, 0.0002, 0.001)))
                        if rng.random() < 0.95:      # 5 %: the tick raised
                            ran = rng.random() < 0.8
                            gate.fs_end(ran=ran)
                            if ran and rng.random() < 0.8:   # else: dropped
                                gate.fs_delivered()
                    elif verdict == FS_DRAIN:
                        gate.fs_drained()
                elif r < 0.78:
                    gate.fs_end(ran=rng.random() < 0.5)
                elif r < 0.82:
                    gate.fs_delivered()
                elif r < 0.87:
                    gate.fs_drained(rng.randrange(3))
                else:
                    time.sleep(rng.choice((0.0, 0.0001, 0.0005)))
                ops[0] += 1
                if ops[0] >= n_ops:
                    done[0].set()
                    if done[1].is_set():
                        break
        except BaseException as e:                               # noqa: BLE001
            errors.append(f"stereo op {ops[0]}: {type(e).__name__}: {e}")
        finally:
            done[0].set()

    def policy():
        rng = random.Random(seed + 1)
        try:
            while time.perf_counter() < give_up:
                r = rng.random()
                if r < 0.14:
                    gate.hold()
                elif r < 0.28:
                    gate.release()
                elif r < 0.42:
                    timeout = rng.choice((None, 0.0, 0.001, 0.004, -1.0))
                    limit = gate.wait_s if timeout is None else max(0.0, timeout)
                    t0 = time.perf_counter()
                    ms, timed_out = gate.wait_idle(timeout)
                    wait_over_s.append(time.perf_counter() - t0 - limit)
                    if not (isinstance(timed_out, bool) and ms >= 0.0):
                        errors.append(f"wait_idle gave {(ms, timed_out)!r}")
                elif r < 0.62:
                    gate.set_active(rng.random() < 0.8)
                elif r < 0.76:
                    got = gate.request_burst(
                        *rng.choice(((), (1,), (2,), (3,), (0,))))
                    if got is not True and got is not False:
                        errors.append(f"request_burst gave {got!r}")
                    if mode == "yield" and got:
                        errors.append("request_burst granted under yield")
                elif r < 0.86:
                    json.dumps(gate.snapshot(), allow_nan=False)
                elif r < 0.88:                       # a whole attempt
                    gate.hold()
                    gate.wait_idle()
                    time.sleep(0.003)                # "the forward"
                    gate.release()
                else:
                    time.sleep(rng.choice((0.0, 0.0002, 0.001)))
                ops[1] += 1
                if ops[1] >= n_ops:
                    done[1].set()
                    if done[0].is_set():
                        break
        except BaseException as e:                               # noqa: BLE001
            errors.append(f"policy op {ops[1]}: {type(e).__name__}: {e}")
        finally:
            done[1].set()

    threads = [threading.Thread(target=stereo, daemon=True),
               threading.Thread(target=policy, daemon=True)]
    for th in threads:
        th.start()
    snaps, last = 0, None
    try:
        while (not (done[0].is_set() and done[1].is_set())
               and time.perf_counter() < give_up + 1.0):
            snap = gate.snapshot()                   # the controller's thread
            json.dumps(snap, allow_nan=False)
            n = snap["counters"]
            if any(v < 0 for v in n.values()) or snap["burst_allowance"] < 0:
                errors.append(f"negative count: {snap!r}")
            if last is not None and any(n[k] < last[k] for k in last):
                errors.append(f"a counter went backwards: {last!r} -> {n!r}")
            last = n
            snaps += 1
            time.sleep(0.001)
    except BaseException as e:                                   # noqa: BLE001
        errors.append(f"controller: {type(e).__name__}: {e}")
    for th in threads:
        th.join(2.0)
    if any(th.is_alive() for th in threads):
        errors.append("a thread did not finish")
    return {"gate": gate, "errors": errors, "begin_s": begin_s,
            "wait_over_s": wait_over_s, "verdicts": verdicts, "ops": ops,
            "snaps": snaps}


def test_property_random_calls_from_two_threads_never_raise_and_fs_begin_never_blocks():
    for mode in ("yield", "only"):
        res = _hammer(mode, seed=20261001, n_ops=2500)
        assert not res["errors"], (mode, res["errors"][:5])
        assert min(res["ops"]) >= 2500, res["ops"]
        assert res["snaps"] > 0
        verdicts = res["verdicts"]
        assert set(verdicts) <= {FS_GO, FS_HOLD, FS_DRAIN}, verdicts
        assert verdicts[FS_GO] > 0 and verdicts[FS_HOLD] > 0, verdicts
        n = res["gate"].snapshot()["counters"]
        assert n["delivered"] > 0, n
        if mode == "yield":
            assert verdicts[FS_DRAIN] == 0           # yield never discards
            assert n["burst_frames"] == 0            # ...nor charges a burst
        else:
            assert verdicts[FS_DRAIN] > 0
            assert n["burst_frames"] > 0             # deliveries met bursts
        assert n["burst_frames"] <= n["delivered"]
        # fs_begin returned at once, every time, whatever the other thread
        # was in the middle of...
        assert len(res["begin_s"]) > 1000
        _assert_at_once(res["begin_s"], f"fs_begin under {mode}")
        # ...and no wait_idle outlasted its bound.
        assert res["wait_over_s"] and max(res["wait_over_s"]) < SLACK_S
        # Whatever state all that left behind: with the policy thread gone
        # every window lapses (the longest here is 6 ms) and FoundationStereo
        # runs.
        gate = res["gate"]
        time.sleep(0.03)
        assert gate.fs_begin() == FS_GO, (mode, gate.snapshot())
        gate.fs_end()


# ------------------------------------- property: random sequences, fake clock
_WINDOWS = dict(lease_s=0.5, inflight_max_s=2.5, heartbeat_s=1.0,
                burst_timeout_s=0.6, starve_s=1.5)

_POLICY_CALLS = (("beat",) * 8 + ("idle",) + ("ask",) * 2 + ("hold",) * 2
                 + ("release",) * 2 + ("wait",))


class _Spec:
    """fs_gate.py's docstrings restated as the least bookkeeping that answers
    "may FoundationStereo start a frame now" — written from their sentences,
    one per method, to be read against them:

      * a hold is a lease: older than lease_s it is ignored;
      * a mission is `only` + set_active(True) no older than heartbeat_s;
      * frames are owed for burst_timeout_s after they were asked for, and
        one is paid when a frame is DELIVERED (fs_delivered) during a mission
        — never at fs_end, which only frees the GPU (and is a frame in the
        record when a map came out of it: ran=True);
      * in a mission with nothing owed, nothing starts — unless nobody has
        asked for starve_s (counted from the last request, or from the
        mission's start);
      * a hold wins over everything, and discards (FS_DRAIN) in a mission;
      * wait_idle — this clock standing still inside it, nothing else
        running — times out exactly when a frame is in flight and not stale.
    """

    def __init__(self, mode, *, lease_s, inflight_max_s, heartbeat_s,
                 burst_timeout_s, starve_s):
        self.mode = mode
        self.lease_s, self.inflight_max_s = lease_s, inflight_max_s
        self.heartbeat_s, self.starve_s = heartbeat_s, starve_s
        self.burst_timeout_s = burst_timeout_s
        self.hold_t = None               # when the policy's hold began
        self.begin_t = None              # when the frame in flight began
        self.active, self.beat_t = False, 0.0
        self.owed, self.ask_t = 0, 0.0   # frames owed to the last request
        self.grant_t = 0.0               # the last request / the mission start
        self.frames = 0                  # maps that came out (fs_end, ran)
        self.delivered = 0               # ...that reached the policy

    def held(self, t):
        return self.hold_t is not None and t - self.hold_t <= self.lease_s

    def flying(self, t):
        return (self.begin_t is not None
                and t - self.begin_t <= self.inflight_max_s)

    def mission(self, t):
        return (self.mode == "only" and self.active
                and t - self.beat_t <= self.heartbeat_s)

    def pending(self, t):
        return self.owed > 0 and t - self.ask_t <= self.burst_timeout_s

    def fs_begin(self, t):
        self.begin_t = None              # a mark left behind is a lost frame
        if self.held(t):
            return FS_DRAIN if self.mission(t) else FS_HOLD
        if (self.mission(t) and not self.pending(t)
                and t - self.grant_t <= self.starve_s):
            return FS_DRAIN
        self.begin_t = t
        return FS_GO

    def fs_end(self, t, ran):
        self.begin_t = None              # the GPU is free; nothing is paid
        if ran:
            self.frames += 1

    def fs_delivered(self, t):
        self.delivered += 1
        if self.mission(t) and self.pending(t):
            self.owed -= 1

    def set_active(self, t, on):
        if self.mode != "only":
            return
        if on:
            if not self.mission(t):
                self.grant_t = t         # a mission starts here
            self.active, self.beat_t = True, t
        else:
            self.active, self.owed = False, 0

    def request_burst(self, t, k):
        if self.mode != "only" or self.pending(t):
            return False
        self.owed, self.ask_t, self.grant_t = k, t, t
        return True


def _walk(mode: str, seed: int, n_ops: int, check: bool = True):
    """A random single-thread sequence on a fake clock. The stereo side keeps
    FStereoWorker.tick's contract — FS_GO, then one fs_end, then (when a map
    came out) fs_delivered or nothing (the tap dropped it) — except that one
    tick in ten "raises" and leaves its mark; its turns interleave with the
    policy side's, so a request can land between a frame's fs_end and its
    fs_delivered, as it can between the two threads. The policy side calls
    anything in any order, wait_idle included (on a live frame too: its bound
    is real time). Time moves mostly by a tick, now and then far enough to
    cross a window. With ``check`` every answer is compared with `_Spec`.
    Returns (gate, clock)."""
    rng = random.Random(seed)
    clock = _Clock(rng.choice((0.0, 7.25, 16000.0)))  # the origin must not matter
    gate = FsGate(mode, clock=clock, **_WINDOWS)
    spec = _Spec(mode, **_WINDOWS)
    trace = []
    stereo = "idle"      # "flying": an FS_GO whose fs_end is still to come;
                         # "ended": a map came out, its delivery still to come

    def same(what, got, want):
        if check:
            assert got == want, (
                f"{mode} seed {seed}: {what} gave {got!r}, the docstring's "
                f"rules give {want!r}; last calls: {trace[-14:]}")

    for _ in range(n_ops):
        r = rng.random()
        if r < 0.80:
            clock.step(rng.choice((0.0, 0.005, 0.02)))
        elif r < 0.94:
            clock.step(rng.uniform(0.0, 0.3))
        elif r < 0.99:
            clock.step(rng.uniform(0.3, 1.2))
        else:
            clock.step(rng.uniform(1.2, 3.5))
        t = clock.t
        if rng.random() < 0.5:                       # the stereo thread's turn
            if stereo == "flying" and rng.random() < 0.9:
                ran = rng.random() < 0.85            # else: no pair, or raised
                gate.fs_end(ran=ran)
                spec.fs_end(t, ran)
                stereo = "ended" if ran else "idle"
                trace.append((round(t, 4), f"fs_end(ran={ran})"))
            elif stereo == "ended":
                if rng.random() < 0.85:              # into the policy mailbox
                    gate.fs_delivered()
                    spec.fs_delivered(t)
                    trace.append((round(t, 4), "fs_delivered"))
                else:
                    trace.append((round(t, 4), "(the tap dropped it)"))
                stereo = "idle"
            else:                # idle — or flying, and that tick raised
                got, want = gate.fs_begin(), spec.fs_begin(t)
                trace.append((round(t, 4), "fs_begin", got))
                same("fs_begin", got, want)
                assert got in (FS_GO, FS_HOLD, FS_DRAIN)
                assert not (mode == "yield" and got == FS_DRAIN)
                stereo = "flying" if got == FS_GO else "idle"
                if got == FS_DRAIN:
                    gate.fs_drained()
        else:                                        # the policy thread's turn
            call = rng.choice(_POLICY_CALLS)
            trace.append((round(t, 4), call))
            if call == "beat":
                gate.set_active(True)
                spec.set_active(t, True)
            elif call == "idle":
                gate.set_active(False)
                spec.set_active(t, False)
            elif call == "ask":
                k = rng.choice((1, 2, 2, 2, 3))
                got, want = gate.request_burst(k), spec.request_burst(t, k)
                trace[-1] = (round(t, 4), f"ask({k})", got)
                same(f"request_burst({k})", got, want)
            elif call == "hold":
                gate.hold()
                spec.hold_t = t
            elif call == "release":
                gate.release()
                spec.hold_t = None
            else:                                    # "wait"
                # The fake clock stands still inside the call and nothing
                # else runs, so a live frame is waited on until the REAL
                # bound: only short timeouts are given then (the default,
                # wait_s, only with nothing in flight).
                flying = spec.flying(t)
                timeout = rng.choice((0.0, 0.001) if flying
                                     else (None, 0.0, 0.001))
                ms, timed_out, _ = _wait(
                    gate, *(() if timeout is None else (timeout,)))
                trace[-1] = (round(t, 4), f"wait({timeout})", timed_out)
                same("wait_idle's timed_out", timed_out, flying)
                assert ms >= 0.0
        snap = gate.snapshot()
        same("snapshot()['held']", snap["held"], spec.held(t))
        same("snapshot()['in_flight']", snap["in_flight"], spec.flying(t))
        same("snapshot()['mission_active']", snap["mission_active"],
             spec.mission(t))
        n = snap["counters"]
        same("counters['frames']", n["frames"], spec.frames)
        same("counters['delivered']", n["delivered"], spec.delivered)
        if spec.pending(t):
            # While frames are owed, the allowance is exactly what is owed.
            same("snapshot()['burst_allowance']", snap["burst_allowance"],
                 spec.owed)
    return gate, clock


def test_property_random_sequences_follow_the_documented_rules():
    for mode in ("yield", "only"):
        seen = Counter()
        for seed in range(60):
            gate, _ = _walk(mode, seed, 400)
            seen.update(_n(gate))
        # The walk is only worth its name if it crossed every window.
        must = ["frames", "delivered", "held_polls", "holds", "lease_expired",
                "waits", "wait_timeouts", "inflight_reset"]
        if mode == "only":
            must += ["drained", "bursts", "burst_frames", "burst_void",
                     "starve_free", "activations"]
        assert all(seen[k] > 0 for k in must), (mode, dict(seen))


def test_property_a_silent_policy_always_reopens_the_gate():
    """NOTHING HERE CAN STOP FoundationStereo FOR GOOD: whatever was called
    before, once the policy thread stops calling (a dead thread, a tick that
    raises every time, shutdown order) every stereo tick is allowed again
    within lease_s (`only`: within max(lease_s, heartbeat_s)) — and stays
    allowed. Whether the frames still reach a policy mailbox or not (every
    other seed)."""
    for mode in ("yield", "only"):
        for seed in range(40):
            gate, clock = _walk(mode, 1000 + seed, 150, check=False)
            deliver = seed % 2 == 0
            quiet = (gate.lease_s if mode == "yield"
                     else max(gate.lease_s, gate.heartbeat_s))
            t_silent = clock.t
            late = []
            while clock.t - t_silent < quiet + 1.0:  # 5 ms stereo ticks
                age = clock.t - t_silent
                verdict = gate.fs_begin()
                if age > quiet:
                    late.append(verdict)
                if verdict == FS_GO:
                    clock.step(0.03)
                    gate.fs_end()
                    if deliver:
                        gate.fs_delivered()
                clock.step(0.005)
            assert late and all(v == FS_GO for v in late), (
                mode, seed, deliver, Counter(late), gate.snapshot())


def test_property_only_heartbeat_without_requests_reopens_after_starve_s():
    """The other way a requester can be stuck: its tick still runs (the
    heartbeat is alive) but it never asks again and never releases. The
    lease, the burst timeout and the starve watchdog between them leave
    FoundationStereo free after starve_s — whether the frames it runs are
    delivered (and spend what is left of an allowance) or not (every other
    seed)."""
    for seed in range(40):
        gate, clock = _walk("only", 2000 + seed, 150, check=False)
        deliver = seed % 2 == 0
        gate.set_active(True)
        quiet = max(gate.starve_s, gate.lease_s, gate.burst_timeout_s)
        t_silent = clock.t
        late = []
        while clock.t - t_silent < quiet + 1.0:
            age = clock.t - t_silent
            verdict = gate.fs_begin()
            if age > quiet:
                late.append(verdict)
            if verdict == FS_GO:
                clock.step(0.03)
                gate.set_active(True)
                gate.fs_end()
                if deliver:
                    gate.fs_delivered()
            clock.step(0.005)
            gate.set_active(True)
        assert late and all(v == FS_GO for v in late), (
            seed, deliver, Counter(late), gate.snapshot())
        assert gate.snapshot()["mission_active"] is True
        assert _n(gate)["starve_free"] >= 1


# -------------------------------------------------------------------- guard
def _guarded(fn):
    """``fn`` behind `_bounded`: a test that does not come back within
    GUARD_S fails, and its thread is left behind as a daemon, instead of
    hanging the file (script and pytest alike)."""
    @functools.wraps(fn)
    def run():
        _bounded(fn, GUARD_S)
    return run


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        globals()[_name] = _guarded(_fn)
del _name, _fn


# ------------------------------------------------------------------- runner
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
