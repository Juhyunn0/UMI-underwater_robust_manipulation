"""
upstream.py — what the two NVlabs checkouts share when they live in ONE process.

FoundationStereo (``--fstereo``) and FoundationPose (``--pose``) are separate
repositories that were each written to be the only thing on ``sys.path``.
Both have a top-level ``Utils.py`` and import it BARE, at module import time:
``core/extractor.py:17`` on the stereo side, ``estimater.py:10`` and eight more
files on the pose side. Whoever imports first owns ``sys.modules["Utils"]``
and the other silently gets the wrong module — FoundationPose star-imports
it, so it would NameError on 31 of its own helpers at the first registration.

The two sessions therefore take this lock around their upstream imports and
never leave the bare name pointing at their own module afterwards (see
``fstereo._import_upstream`` and ``session.PoseSession._bring_up_fp``).
"""

from __future__ import annotations

import threading

#: Serialises the import of one upstream checkout against the other's, so the
#: pop-and-restore of ``sys.modules["Utils"]`` in one cannot interleave with a
#: bare ``from Utils import`` in the other. Re-entrant: a session may import
#: twice (a restart after close()).
UPSTREAM_IMPORT_LOCK = threading.RLock()

#: Where FoundationStereo's own ``Utils`` module is kept reachable after its
#: import, since it must not stay under the bare name.
FSTEREO_UTILS_ALIAS = "fstereo_upstream.Utils"

#: Serialises the HEAVY model loads (FoundationStereo 3 GiB + CUDA context,
#: the diffusion-policy checkpoint + its warm-up) against each other. Two
#: loads on two threads at the same moment as the acados solver build
#: crashed the process twice on 2026-09-02 — once "the monitored command
#: dumped core" during 'DP policy: loading selected.ckpt' (policy_dryrun,
#: 1 of 4 launches), once "malloc(): invalid size (unsorted)" at t=2.4 s of
#: rov_gui/tools/policy_bench_check.py (JSON written, process hung at exit;
#: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_150726.json). The
#: cause is not attributed (torch.load + dlopen of the freshly built solver
#: + a second torch.load in one process); the mitigation is to never run
#: two of these loads concurrently. Held for the WHOLE load, not just the
#: import (UPSTREAM_IMPORT_LOCK above covers the sys.modules dance only).
MODEL_LOAD_LOCK = threading.Lock()
