#!/usr/bin/env python3
"""
runstore.py — one dated folder per run, shared by everything that records.

Before this module the station scattered a single pool run across three trees:
the screen recording in ``sessions/ui_recordings``, the raw localization in
``sessions/nav_runs/<stamp>/``, the controller CSV in ``sessions/mpc_runs``.
One press of START produced ``ui_20260814_184149.mp4``,
``nav_runs/20260814_184148/`` and ``mpc_runs/20260814_184151.csv`` — the same
run, three places, related only by a timestamp the reader had to match by eye.

Now every writer asks this module where to put its file and they all land
together::

    data/20260814/0814_184148/
        ui_20260814_184149_squaretest.mp4     the pilot's view (+ .json sidecar)
        mission_log.txt                        what the log said, wall-clock
        nav_184148/                            map.json fixes.csv detections.csv
        mpc_184151.csv  mpc_184151.meta.json   the controller's own record
        controller.json                        M / C / D + the run's gains
        events.log                             engage / refusal / disengage

ONE root, dated (2026-09-14)
----------------------------
``data/`` is the only root. It used to be two — ``data/`` for the gantry /
ZED / survey captures and training outputs, ``sessions/`` for everything the
station wrote, and inside ``sessions/`` nine differently named trees
(``low_level_controller_data``, ``policy_observe``, ``land_dryruns``,
``dryruns``, ``nav_runs``, ``mpc_runs``, ``ui_recordings``, ``pose_meshes``,
``demonstration_NNNN``). Operator request 2026-09-14: one folder, by date.
``tools/migrate_data_layout_20260914.py`` moved every old folder and
``data/MIGRATION_20260914.json`` maps each old path to its new one.

The leaf name: ``MMDD_HHMMSS[_kind]``
-------------------------------------
Operator requests, 2026-08-14, in two steps. First the date: a run folder is
dragged out of its date directory about as often as it is opened in place —
into a message, a plot script's argument, a backup — and ``1841`` alone says
nothing once it has moved. Then the seconds: a pool session produces several
runs a minute, and minute resolution silently merged them.

The KIND suffix (2026-09-14) is what the separate trees used to be. A water
run has none (``0913_155559``); every other kind of record carries one so a
reader — and :func:`run_dir`'s join below — can tell them apart::

    0913_155559            water run: the station flying (or holding) in the pool
    0913_155559_observe    LOW level None / --policy-observe: the policy ran,
                           nothing commanded, the pilot dragged the vehicle
    0903_134502_landdry    --land-dry-run: the station on a bench, in air
    0907_132422_dryrun     rov_gui.tools.policy_dryrun: synthetic plant
    0823_162647_obj        an on-site object reconstruction (--pose)

The tree used to be "a safety property of the RECORD, not a tidiness
preference" (MpcWorker._run_tree): a bench run started inside the join
window of a water run would have landed in ITS folder and appended to the same
``plans.jsonl``, and nothing in the meta told them apart afterwards. The kind
keeps that guarantee in the leaf name instead of the path above it —
:func:`run_dir` only ever joins a folder of the SAME kind, and
:func:`leaf_kind` is how a scanner (plot_runs, the policy tools) keeps a
``_observe`` or ``_landdry`` record out of a water statistic.

Folders written before those changes keep their old ``HHMM`` name and are NOT
renamed, for the same reason the migration renamed nothing inside a folder:
a path that moves turns a citation into a dead reference, and the migration
already cost one round of rewriting. The consequence to know about is that the
join below only ever joins folders in the CURRENT naming — an old ``1841/``
is never reopened by a new run, which is the safe direction.

Why the folder is FOUND, not computed, and why that changed
-----------------------------------------------------------
The writers do not know about each other and must not have to. The screen
recorder is a GUI-thread object, the nav recorder is the window, the CSV
belongs to the MpcWorker on its own thread — coordinating a shared run id
across those means a handshake, and a handshake has a failure mode (one writer
misses it and its file goes somewhere else, silently).

While the leaf was a MINUTE, that needed no lookup: three writers asking
independently at 18:41:48, :49 and :51 all *computed* ``1841``, and a join
window was needed only for the run that straddled a minute boundary.

**Seconds resolution removes that property**, and this is the trade the second
request bought: three writers seconds apart now compute three different names,
so :func:`run_dir` no longer computes the answer — it LOOKS for the run already
in progress in today's directory and joins it, and only starts a new folder
when there is none. The filesystem is still the only state and there is still
no handshake, but :data:`JOIN_WINDOW_S` went from covering one edge case to
being what holds every run together. If a writer ever starts more than
:data:`JOIN_WINDOW_S` after the one before it, its files land in a folder of
their own — the same thing that happened before across a minute boundary, just
reachable from more directions.

Merging only. Nothing here ever splits or moves a file that already landed.
``join=False`` is the one way to refuse a join: it starts a fresh folder even
while one is open, for a caller that must own its folder (a dryrun tool that
runs twice a minute), and the writers that follow it join THAT one, because
it is the newest.

A folder is a RUN, not a launch
-------------------------------
Anything that calls :func:`run_dir` creates a folder, so a single line written
for a reason other than flying leaves one behind. Seven of those accumulated on
2026-08-14, each holding only ``MpcWorker.setup()``'s build fingerprint — the
station had been started and nothing was ever flown. That line is now deferred
(``MpcWorker._log_event(defer=True)``) and heads up each run folder's
``events.log`` instead. Anything else that wants to write outside a run should
do the same rather than opening a folder to say it exists.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import NamedTuple

#: Everything a run produces goes under here. One tree, dated.
DEFAULT_BASE = Path("data")


class Tree(NamedTuple):
    """Where a writer's run folder goes: the root plus the run's KIND.

    Passed to :func:`run_dir` in place of a bare base path by the one place
    that knows the kind (``MpcWorker._run_tree``), and handed on unchanged by
    the writers that follow the controller (SlamNavWorker, the depth
    recorder). A bare ``str``/``Path`` base still means kind "" (water).
    """
    base: str
    kind: str = ""

#: How long after the LAST file was created in a run folder another writer
#: still counts as belonging to that same run. 90 s comfortably covers one
#: START's writers (they are seconds apart) and is far short of the gap
#: between two runs a pilot would think of as separate.
#:
#: Two clocks answer "when was this folder last touched", and the LATER one
#: wins: the folder's own NAME (when the run began) and its directory mtime
#: (which bumps on file CREATION, not on append — so it measures "a writer
#: opened something here", which is the question being asked). The name alone
#: would end a run 90 s after it began even while writers were still arriving;
#: mtime alone cannot be trusted when a caller passes ``when`` (tests, replay),
#: because the folder was mkdir'd at a real wall-clock time that has nothing to
#: do with the timestamp being asked about.
JOIN_WINDOW_S = 90.0

#: A leaf stamped up to this far in the FUTURE still counts as begun. The only
#: way such a leaf exists is ``run_dir(join=False)`` stepping its stamp past a
#: same-second neighbour; without the slack the writers that follow it inside
#: that second would join the neighbour instead — the very thing it refused.
_FUTURE_SLACK_S = 1.0

#: The current leaf-name shape, ``MMDD_HHMMSS[_kind]``. Deliberately strict:
#: it is what keeps a pre-2026-08-14 ``1841/`` (or the short-lived
#: ``0814_1841``) from ever being joined by a new run, and the kind group is
#: what keeps a ``_observe`` folder from being joined by a water writer.
_LEAF = re.compile(r"^(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:_([a-z][a-z0-9]*))?$")
#: A kind is one lowercase word (``observe``, ``landdry``, ``dryrun``, ``obj``)
#: so the leaf stays parseable — an underscore inside it would read as a
#: second suffix, and an uppercase letter or a dash is not a kind.
_KIND = re.compile(r"^[a-z][a-z0-9]*$")
#: Pre-2026-08-14 water leaves (``2004`` / ``0814_2212``): never joined, but a
#: scanner must still count them as water runs (see :func:`leaf_kind`).
_LEGACY_LEAF = re.compile(r"^\d{4}(?:_\d{4})?$")


def run_dir(base: "Tree | Path | str" = DEFAULT_BASE, when: float | None = None,
            create: bool = True, kind: str = "", join: bool = True) -> Path:
    """``<base>/YYYYMMDD/MMDD_HHMMSS[_kind]`` — where this moment's files belong.

    The run of the SAME kind already in progress in today's directory if
    there is one (see :data:`JOIN_WINDOW_S`), otherwise a new folder stamped
    now. Two threads — or two processes — asking within the same run get the
    same answer without talking to each other; the filesystem is the only
    shared state. ``base`` may be a :class:`Tree`, whose kind wins over the
    ``kind`` argument.

    ``create=False`` is for inspection only. With seconds in the name the
    answer is no longer a pure function of the clock: if nothing is in
    progress, two callers a second apart get two different (uncreated) paths,
    where the folder they actually agree on is the one the first of them
    creates.

    ``join=False`` starts a fresh folder even when one is open (it is then
    the newest, so later writers join it) — the per-invocation isolation a
    tool needs when it may run twice inside :data:`JOIN_WINDOW_S`.
    """
    if isinstance(base, Tree):
        base, kind = base.base, (base.kind or kind)
    kind = str(kind or "")
    if kind and not _KIND.match(kind):
        raise ValueError(f"run kind {kind!r} must match {_KIND.pattern}")
    t = time.time() if when is None else float(when)
    day = Path(base) / time.strftime("%Y%m%d", time.localtime(t))
    if join:
        joined = _run_in_progress(day, t, kind)
        if joined is not None:
            return joined
    cur = day / (time.strftime("%m%d_%H%M%S", time.localtime(t))
                 + (f"_{kind}" if kind else ""))
    if not join:
        # A FRESH folder was asked for: the same second must not hand back
        # the one a previous caller just opened, so step the stamp forward
        # (the leaf keeps its shape; a second is nothing next to the run).
        while cur.exists():
            t += 1.0
            cur = day / (time.strftime("%m%d_%H%M%S", time.localtime(t))
                         + (f"_{kind}" if kind else ""))
    if create:
        cur.mkdir(parents=True, exist_ok=True)
    return cur


def leaf_kind(name: str) -> str | None:
    """The kind a run-folder name carries: ``""`` for a water run (current or
    pre-2026-08-14 shape), ``"observe"`` / ``"landdry"`` / … for a suffixed
    one, None for a name that is not a run folder at all (a training run, a
    gantry capture, ``train_logs``). Scanners use this to keep the kinds
    apart now that they share a date directory."""
    m = _LEAF.match(name)
    if m is not None:
        return m.group(6) or ""
    if _LEGACY_LEAF.match(name):
        return ""
    return None


def _leaf_time(day: Path, name: str, kind: str = "") -> float | None:
    """The wall-clock time a leaf name encodes, or None if it is not one —
    or not one of the requested ``kind``."""
    m = _LEAF.match(name)
    if m is None or (m.group(6) or "") != kind:
        return None
    mon, dom, hh, mm, ss = (int(v) for v in m.groups()[:5])
    try:
        year = int(day.name[:4])
    except ValueError:
        return None
    try:
        # DST-ambiguous local times resolve to whichever mktime picks; a run
        # folder only needs to compare against its own neighbours, so an hour
        # of ambiguity once a year costs at most one extra folder.
        return time.mktime((year, mon, dom, hh, mm, ss, 0, 0, -1))
    except (OverflowError, ValueError):
        return None


def _run_in_progress(day: Path, t: float, kind: str = "") -> Path | None:
    """The run folder of ``kind`` that is still open at ``t``, or None to
    start a new one.

    Newest first, so a directory holding a whole session's folders costs one
    scan and never picks an older run over a newer one.
    """
    best: Path | None = None
    best_seen: float | None = None
    try:
        entries = list(day.iterdir())
    except OSError:
        return None
    for d in entries:
        started = _leaf_time(day, d.name, kind)
        if started is None or started > t + _FUTURE_SLACK_S or not d.is_dir():
            continue                     # not a run folder, or not yet begun
        try:
            touched = d.stat().st_mtime
        except OSError:
            touched = started
        # See JOIN_WINDOW_S: mtime wins when it is real, the name wins when the
        # caller is asking about a `when` the filesystem knows nothing about.
        seen = started if touched > t else max(started, touched)
        if best_seen is None or seen > best_seen:
            best, best_seen = d, seen
    if best is not None and (t - best_seen) <= JOIN_WINDOW_S:
        return best
    return None


def stamp(fmt: str = "%Y%m%d_%H%M%S", when: float | None = None) -> str:
    """Wall-clock stamp for a FILE name (the folder carries the run's start)."""
    return time.strftime(fmt, time.localtime(
        time.time() if when is None else float(when)))


# A recording name is typed by a pilot mid-session, so it arrives with spaces,
# slashes, Korean, whatever. Two things must survive that: the file must open
# on this filesystem, and ``path.with_suffix(".json")`` must still produce the
# sidecar rather than eating part of the name at a dot.
_UNSAFE = re.compile(r"[^0-9A-Za-z가-힣_\-]+")


def slug(name: str, max_len: int = 48) -> str:
    """Operator text -> a filename fragment. Empty string if nothing survives.

    Keeps letters (including Hangul), digits, ``_`` and ``-``; every run of
    anything else — dots included — collapses to a single underscore.
    """
    out = _UNSAFE.sub("_", str(name or "").strip()).strip("_")
    return out[:max_len].strip("_")
