#!/usr/bin/env python3
"""rl_policy_library.py — fill and list the RL POLICY LIBRARY (control/rl_library.py; 2026-11-01).

    ./c3 rl-policies sync --dry-run              # what would come in from RL_controller (writes NOTHING)
    ./c3 rl-policies sync                        # copy the new exports into data/rl_policies/
    ./c3 rl-policies list                        # every export some RL LOW can fly (station + library)
    ./c3 rl-policies list --low rl_pwm_7d        # exactly what the LOW row's picker offers RL_PWM_7D
    ./c3 rl-policies list --all                  # + the no-fly / unflyable / smoke-length / unreadable ones, with why

(= ~/miniforge3/envs/robust/bin/python rov_gui/tools/rl_policy_library.py ..., from any cwd: a RELATIVE --dst /
--station / --rl-root is REPO-ROOT-relative, exactly as a policy_dir is — rl_library.resolve_dir.)

Operator request 2026-11-01: "RL 훈련한 것들을 어떤 폴더에 다 모아두고" — every RL_controller export the station could
fly, gathered in ONE git-ignored folder (data/rl_policies/, operator decision) that the LOW row's picker lists beside
the git-tracked station folder rov_gui/control/rl_policies/ (which this tool only ever READS). `sync` refuses a --dst
that is (inside) the station folder, inside RL_controller, or inside this repo anywhere but its git-ignored data/ —
checked on the paths as given AND on their realpaths (2026-11-02: a symlinked --dst / data/ / --rl-root alias passed
the text-only checks, and install() would have written through the link into the target).

SOURCES (RL_controller, READ ONLY — this tool never writes there; --dst inside it is refused):
  <root>/logs/rsl_rl/rov_posehold/<YYYY-MM-DD_HH-MM-SS>_<tag>/exported/   (scripts/export_policy.py's output)
  <root>/Policies/<name>/                                                  (hand-placed exports, e.g. Josh's
                                                                            pwm_k0_s0; dirs with obs_spec.json)
<root> = --rl-root, else $RL_CONTROLLER_ROOT, else ~/Desktop/RL_controller.

WHY COPIES, never links into RL_controller: its export folders change IN PLACE — traj10_s1's obs_spec.json was
re-exported 2026-09-27 (sha1 efba0d34fcff vs the station copy's da3759dc8e1d, rl_library.py docstring), Isaac
play.py overwrites exported/policy.onnx (RL_controller scripts/play.py:176), and a same-tag rerun reuses the name
(RL_controller scripts/_pipeline_cfdab_seed.sh:23 would name the cfd_drag_v2 seed-1 run pwm11att14Rcfd_s1 again —
the v1 export of that name was flown 2026-11-01). A picker that read the live folders would fly other bytes behind an
unchanged name.

SKIPPED, each with its reason (counted in the summary; nothing is ever imported "with a warning"):
  * a folder starting '_' or '.' (_aborted_20261026_reboot holds the 10-26 first attempts, no export);
  * no run stamp / no tag (rsl_rl's unnamed 2-iteration smoke folders, e.g. 2026-09-21_13-31-02);
  * a tag starting "smoke" — 4 smoke runs DO have complete-looking exports that would pass the station's layout
    checks (RL_controller logs/rsl_rl/rov_posehold/*_smoke_{att7_axes6,pwm11,pwm11att,r2smokeT}/exported, obs_spec
    rsl_rl_iteration 2/2/2/1, read 2026-11-01);
  * no exported/ (aborted first attempts), or obs_spec.json / policy_weights.npz missing;
  * any file of the export modified < --settle-s ago: export_policy.py writes npz, onnx, onnx, obs_spec IN THAT
    ORDER (:341, :344, :411, :488-489), so an export that is still being written looks like a whole one;
  * (run folders only) obs_spec.json OLDER than policy_weights.npz — that write order again: an in-place re-export
    that died between the two (an exception or a kill after np.savez) leaves the NEW weights beside the OLD obs_spec,
    whose checkpoint / rsl_rl_iteration would then label the wrong weights, and the settle window stops seeing it
    120 s later (2026-11-02). Policies/ hand copies are exempt: their obs_spec is 42-67 ms older than their npz
    [측정: os.stat of RL_controller Policies/pwm_k0.04_s0 and pwm_k0_s0, read 2026-11-02 — 42 / 67 ms].
    What no rule here can see: a parity assert that fails AFTER obs_spec.json is written (:491 integral twin, :509
    wrench head, :513 numpy/torch) leaves complete, correctly ordered files;
  * obs_spec.json not valid JSON; rsl_rl_iteration absent or < --min-iteration (the smoke guard behind the tag rule;
    default rl_library.MIN_ITERATION — the picker's own floor, below which it never offers an export anyway);
  * policy_weights.npz the station's OWN loader (rl_policy.NumpyPolicy) cannot load — loaded from the very bytes that
    are hashed, so what was checked is what gets copied;
  * inconsistent dims: obs_dim != frame_dim x history_len, first-layer input width != frame_dim x history_len (the
    station's build check), last-layer width != num_actions, layers that do not chain;
  * non-finite weights (NaN / inf, an obs_var <= 0 the station would divide by) or a non-finite action on a zero
    observation — a diverged run must not reach a picker.
A flight-banned export (rl_library.NO_FLY) or one no RL LOW flies is NOT skipped (DESIGN 2026-11-01 §5): it is
imported, the picker shows it greyed, and the worker refuses it at the pick — by name, by weights sha1 and by the run
its obs_spec checkpoint names (rl_library.no_fly_reason). The sync output marks both.

NAMES: the run tag (the Policies folder name). A name is IMMUTABLE — a past run meta's policy_dir must keep meaning the
same bytes — so an entry is NEVER overwritten, modified or deleted. If <name> is already held — by the library
(--dst), by the station folder (--station, read only), or by an earlier source of this same sync — with the same
(obs_spec.json sha1, policy_weights.npz sha1): "up to date"; with other bytes: '<name>@<YYYYMMDD-HHMMSS>' (the run
folder's stamp; Policies: obs_spec.json's mtime), same check; both held by other bytes: skipped. "Up to date" looks at
the whole tag FAMILY — <name> and every '<name>@*' already held — not only at those two names (2026-11-02: a Policies/
stamp moves with a touch / a git checkout / a cp without -p, and the same bytes came in again under a new '@stamp');
the naming of new bytes is unchanged. The station folder is
part of that namespace (2026-11-01: its pwm12PI_s0 and traj10_s1 differ from RL_controller's same-tag exports — a
library copy under the same name would put two different exports called "pwm12PI_s0" in front of the operator), and
its byte-identical exports are not copied again (they are listed once anyway: rl_library.dedupe keeps the station
copy). Sources are processed oldest run first, so a re-used tag's first run keeps the plain name.

AN IMPORT: obs_spec.json, policy_weights.npz and, when present, policy.onnx, policy_wrench.onnx, README*.md (never
policy.pt, model_*.pt, params/) are copied into <dst>/.tmp-<name>-<pid>/ (hidden: rl_library.scan skips it), re-hashed
(they must be the bytes that were validated), LIBRARY.json is written (rl_library.SIDECAR: provenance + the two sha1s
scan() verifies on every read), every file is chmod 0444, fsync'd, and the folder is renamed to <dst>/<name> (atomic;
refused if the name appeared meanwhile — on Linux rename() would silently REPLACE an empty directory, so lexists() is
checked first and a non-empty one makes rename() fail). Any failure removes the temp folder and leaves the library as
it was; the run's exit code is 1.

NO Qt (the CLI and its tests run headless), numpy only for the weights check. RUN BY PATH, never `python -m
rov_gui.tools...`: RL_controller's GPU guard SIGSTOPs guarded training while `pgrep -f '^[^ ]*python[0-9.]* (.* )?-m
rov_gui'` matches (RL_controller scripts/_gpu_guard.sh:13) — and a sync is exactly what runs WHILE training.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import re
import shutil
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import rl_library as L                      # noqa: E402

# ----------------------------------------------------------------------------- where things are
#: rsl_rl's log dir for the rov_posehold task, under the RL_controller root (RL_controller scripts/train.py)
RUNS_SUBDIR = os.path.join("logs", "rsl_rl", "rov_posehold")
#: hand-placed exports (2026-10-15: Josh's pwm_k0_s0 / pwm_k0.04_s0; their source runs are not on this machine)
POLICIES_SUBDIR = "Policies"
RL_ROOT_ENV = "RL_CONTROLLER_ROOT"
DEFAULT_RL_ROOT = "~/Desktop/RL_controller"
#: copied besides REQUIRED_FILES when present: what RL_controller's deploy tools read (the station reads only the
#: two required files) and the export's own notes. Never policy.pt (Isaac play.py's TorchScript) or anything else.
OPTIONAL_FILES = ("policy.onnx", "policy_wrench.onnx")
_README_RE = re.compile(r"^README[^/]*\.md$")
SIDECAR_SCHEMA = 1
#: the temp folder an import is assembled in; '.'-prefixed so rl_library.scan never lists a half-copied entry
TMP_PREFIX = ".tmp-"

# ----------------------------------------------------------------------------- outcomes (summary order)
SKIP_NOT_RUN = "folder starts with '_' or '.'"
SKIP_NO_STAMP = "no run stamp in the folder name"
SKIP_NO_TAG = "unnamed run (no tag)"
SKIP_SMOKE = "smoke run"
SKIP_NO_EXPORT = "no exported/"
SKIP_MISSING = "obs_spec.json / policy_weights.npz missing"
SKIP_SETTLING = "still being written"
SKIP_INTERRUPTED = "export interrupted / re-export in progress"
SKIP_JSON = "obs_spec.json unreadable / not JSON"
SKIP_ITERATION = "rsl_rl_iteration below --min-iteration"
SKIP_NPZ = "policy_weights.npz unreadable"
SKIP_DIMS = "inconsistent dims"
SKIP_NONFINITE = "non-finite weights"
SKIP_CHANGED = "export changed while it was read"
SKIP_NAME = "name held by other bytes"
SKIP_ORDER = (SKIP_NOT_RUN, SKIP_NO_STAMP, SKIP_NO_TAG, SKIP_SMOKE, SKIP_NO_EXPORT, SKIP_MISSING, SKIP_SETTLING,
              SKIP_INTERRUPTED, SKIP_JSON, SKIP_ITERATION, SKIP_NPZ, SKIP_DIMS, SKIP_NONFINITE, SKIP_CHANGED,
              SKIP_NAME)
IMPORT, UP_TO_DATE, SKIP, FAILED = "import", "up to date", "skip", "failed"


@dataclass
class Source:
    """One candidate export folder of the RL_controller tree."""

    path: str                           # the export dir, absolute (rl_library.resolve_dir form, no trailing '/')
    tag: str                            # the run tag, or the Policies/ folder name: the library name before any '@'
    stamp: _dt.datetime | None          # the '@stamp': the run folder's stamp; Policies/: obs_spec.json's mtime
    stamp_from: str                     # "run folder" | "obs_spec mtime" (the sidecar says which)
    run: str | None                     # the rsl_rl run folder's NAME (logs/), None for Policies/
    run_dir: str | None                 # ... and its absolute path (params/env.yaml lives there)


@dataclass
class Item:
    """One decision of a sync (or a dry run): import | up to date | skip | failed."""

    source: str                         # the folder looked at (absolute)
    tag: str                            # what it is called before any '@stamp' (the folder name when it has no tag)
    action: str
    reason: str = ""                    # a SKIP_* constant for a skip; "" otherwise
    detail: str = ""                    # the specifics: which file / value, the copy it equals, the error
    name: str = ""                      # the library name (import / up to date)
    obs_sha1: str = ""
    npz_sha1: str = ""
    spec: dict = field(default_factory=dict, repr=False)
    src: Source | None = None
    entry: L.Entry | None = None        # rl_library's own reading of the SOURCE (slot fit, no-fly, cap, ...)
    held_by: str = ""                   # up to date: the repo-relative (else absolute) folder holding the same bytes

    @property
    def id(self) -> str:
        return L.export_id(self.obs_sha1, self.npz_sha1) if self.obs_sha1 and self.npz_sha1 else ""


# ----------------------------------------------------------------------------- discovery (folder-level rules)
def default_rl_root() -> str:
    """$RL_CONTROLLER_ROOT, else ~/Desktop/RL_controller — resolved like a policy_dir (rl_library.resolve_dir)."""
    return L.resolve_dir(os.environ.get(RL_ROOT_ENV) or DEFAULT_RL_ROOT)


def _subdirs(d: str) -> list[str] | None:
    """Sorted names of the directories in d (symlinks to directories count); None when d cannot be listed."""
    try:
        with os.scandir(d) as it:
            return sorted(e.name for e in it if e.is_dir())
    except OSError:
        return None


def discover(rl_root: str) -> tuple[list[Source], list[Item]]:
    """(candidate export folders, folder-level skips) under an RL_controller root, runs oldest first (the run folder
    names sort chronologically), then Policies/ by name. Loose files are not runs and are not mentioned."""
    sources, skipped = [], []
    runs_dir = os.path.join(rl_root, RUNS_SUBDIR)
    for name in _subdirs(runs_dir) or []:
        d = os.path.join(runs_dir, name)
        if name.startswith(("_", ".")):
            skipped.append(Item(d, name, SKIP, SKIP_NOT_RUN))
            continue
        tag, stamp = L.checkpoint_run(name)             # '<YYYY-MM-DD_HH-MM-SS>[_<tag>]' -> (tag | None, stamp | None)
        if stamp is None:
            skipped.append(Item(d, name, SKIP, SKIP_NO_STAMP))
            continue
        if not tag:
            skipped.append(Item(d, name, SKIP, SKIP_NO_TAG))
            continue
        if tag.startswith("smoke"):
            skipped.append(Item(d, tag, SKIP, SKIP_SMOKE))
            continue
        ex = os.path.join(d, "exported")
        if not os.path.isdir(ex):
            skipped.append(Item(d, tag, SKIP, SKIP_NO_EXPORT))
            continue
        sources.append(Source(ex, tag, stamp, "run folder", name, d))
    pol_dir = os.path.join(rl_root, POLICIES_SUBDIR)
    for name in _subdirs(pol_dir) or []:
        d = os.path.join(pol_dir, name)
        if not os.path.isfile(os.path.join(d, L.REQUIRED_FILES[0])):
            continue                                    # Policies/ also holds hw_mpc_josh_*.yaml etc.: not exports
        if name.startswith(("_", ".")):
            skipped.append(Item(d, name, SKIP, SKIP_NOT_RUN))
            continue
        if name.startswith("smoke"):
            skipped.append(Item(d, name, SKIP, SKIP_SMOKE))
            continue
        try:
            stamp = _dt.datetime.fromtimestamp(os.stat(os.path.join(d, L.REQUIRED_FILES[0])).st_mtime)
            stamp = stamp.replace(microsecond=0)
        except (OSError, OverflowError, ValueError):
            stamp = None
        sources.append(Source(d, name, stamp, "obs_spec mtime", None, None))
    return sources, skipped


# ----------------------------------------------------------------------------- inspection (file-level rules)
def _mtime(ns: int) -> str:
    """A file mtime (st_mtime_ns) as local 'YYYY-MM-DD HH:MM:SS.mmm' — the interrupted-export skip names both."""
    return _dt.datetime.fromtimestamp(ns / 1e9).isoformat(sep=" ", timespec="milliseconds")


def _newest_file(d: str) -> tuple[float, str]:
    """(mtime, name) of the most recently modified file in d; (-inf, "") when it holds none. OSError passes."""
    newest, who = float("-inf"), ""
    with os.scandir(d) as it:
        for e in it:
            if e.is_file():
                m = e.stat().st_mtime
                if m > newest:
                    newest, who = m, e.name
    return newest, who


def check_weights(spec: dict, npz_bytes: bytes) -> tuple[str, str]:
    """("", "") when the weights are fit to fly as the spec describes them, else (SKIP_NPZ | SKIP_DIMS |
    SKIP_NONFINITE, detail). Loaded by the station's OWN loader (rl_policy.NumpyPolicy: n_layers, W<i>/b<i>,
    activation, optional obs_mean/obs_var — RL_controller scripts/export_policy.py:335-341 writes them) from the bytes
    that were hashed: what it cannot load, the station cannot fly. The dims are the station build's own check
    (HwRl.__init__: policy obs_dim == frame_dim x history_len, act_dim == num_actions) plus the spec's obs_dim /
    act_dim and the layer chain the forward pass needs."""
    import numpy as np                                  # here, not at import: `list` never loads numpy
    from rov_gui.control.rl_policy import NumpyPolicy

    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            # np.load takes a file object as well as a path: the policy is built from the very bytes hashed above
            pol = NumpyPolicy(io.BytesIO(npz_bytes))
    except Exception as e:                              # noqa: BLE001 — any failure to load is "unreadable"
        return SKIP_NPZ, f"{type(e).__name__}: {e}"[:160]
    try:
        k, fd, na = (int(spec[x]) for x in ("history_len", "frame_dim", "num_actions"))
    except (KeyError, TypeError, ValueError):
        return SKIP_DIMS, "obs_spec has no integer history_len / frame_dim / num_actions"
    width = fd * k
    prev = None
    for i, (w, b) in enumerate(pol.layers):
        if w.ndim != 2 or b.shape != (w.shape[0],) or (prev is not None and w.shape[1] != prev):
            return SKIP_DIMS, f"layer {i} (W{i} {tuple(w.shape)}, b{i} {tuple(b.shape)}) does not chain"
        prev = w.shape[0]
    for key, want, what in (("obs_dim", width, f"frame_dim {fd} x history_len {k}"),
                            ("act_dim", na, "num_actions")):
        if key in spec:
            try:
                v = int(spec[key])
            except (TypeError, ValueError):
                return SKIP_DIMS, f"obs_spec {key} {spec[key]!r} is not an integer"
            if v != want:
                return SKIP_DIMS, f"obs_spec {key} {v} != {what} = {want}"
    if pol.obs_dim != width:
        return SKIP_DIMS, f"first layer takes {pol.obs_dim} inputs != frame_dim {fd} x history_len {k} = {width}"
    if pol.act_dim != na:
        return SKIP_DIMS, f"last layer gives {pol.act_dim} outputs != num_actions {na}"
    for key, v in (("obs_mean", pol.obs_mean), ("obs_var", pol.obs_std)):
        if v is not None and tuple(v.shape) != (pol.obs_dim,):
            return SKIP_DIMS, f"{key} shape {tuple(v.shape)} != ({pol.obs_dim},)"
    arrays = [a for w, b in pol.layers for a in (w, b)] + [a for a in (pol.obs_mean, pol.obs_std) if a is not None]
    if not all(bool(np.isfinite(a).all()) for a in arrays):
        return SKIP_NONFINITE, "NaN / inf in the weights or the normalizer (a diverged run?)"
    if pol.obs_std is not None and not bool((pol.obs_std > 0.0).all()):
        return SKIP_NONFINITE, "obs_var <= 0 somewhere — the station divides by its square root"
    # NumpyPolicy.act clips to [-1, 1], which turns an overflow (inf) into a silent full command: make numpy RAISE
    # on overflow / invalid / divide instead (underflow stays quiet: ELU's expm1 of a very negative input is fine)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            y = pol.act(np.zeros(pol.obs_dim))
    except FloatingPointError as e:
        return SKIP_NONFINITE, f"the forward pass on a zero observation fails ({e})"
    if not bool(np.isfinite(y).all()):
        return SKIP_NONFINITE, "a zero observation gives a non-finite action"
    return "", ""


def _entry_for(path: str) -> L.Entry | None:
    """rl_library's own reading of one export folder (slot fit, no-fly, cap, trained, ...) — through its public scan()
    of the parent, so the sync shows exactly what the picker will show and never re-implements how it reads one."""
    for e in L.scan([("other", os.path.dirname(path))]):
        if e.path == path:
            return e
    return None


def inspect_export(src: Source, min_iteration: int, settle_s: float, now: float) -> Item:
    """The file-level rules for one candidate (module docstring, SKIPPED): an IMPORT candidate (no name yet) or a
    skip. Reads only."""
    d = src.path

    def skip(reason, detail="", **kw) -> Item:
        return Item(d, src.tag, SKIP, reason, detail, src=src, **kw)

    missing = [f for f in L.REQUIRED_FILES if not os.path.isfile(os.path.join(d, f))]
    if missing:
        return skip(SKIP_MISSING, "no " + " / ".join(missing))
    try:
        newest, who = _newest_file(d)
    except OSError as e:
        return skip(SKIP_MISSING, f"cannot list the folder ({e.strerror or e})")
    age = now - newest
    if age < settle_s:                      # a future mtime (clock skew) is negative: "being written" (fail-closed)
        return skip(SKIP_SETTLING, f"{who} modified {max(age, 0.0):.0f} s ago (< --settle-s {settle_s:g})")
    if src.run is not None:
        # 2026-11-02 (review): the settle window only sees an export that is STILL being written. export_policy.py
        # writes policy_weights.npz FIRST and obs_spec.json LAST (RL_controller scripts/export_policy.py:341 vs
        # :488-489, the onnx exports and the pwm mixer assert :368 in between), so an in-place re-export that died
        # in between leaves NEW weights beside the OLD obs_spec — whose checkpoint / rsl_rl_iteration would then label
        # them in the picker, the sidecar and the run meta. Strict '<': every finished run export has obs_spec.json
        # newer than its npz [측정: 182 of 182, 0 equal — os.stat of RL_controller logs/rsl_rl/rov_posehold/*/exported/
        # {obs_spec.json,policy_weights.npz}, read 2026-11-02]. Run folders only: the Policies/ hand copies were not
        # written by the exporter (module docstring). A crashed re-export of the SAME checkpoint is skipped too —
        # fail-closed; re-running the export clears it.
        try:
            t_obs, t_npz = (os.stat(os.path.join(d, f)).st_mtime_ns for f in L.REQUIRED_FILES)
        except OSError as e:
            return skip(SKIP_MISSING, f"cannot stat ({e.strerror or e})")
        if t_obs < t_npz:
            return skip(SKIP_INTERRUPTED, f"obs_spec.json modified {_mtime(t_obs)}, BEFORE policy_weights.npz "
                                          f"{_mtime(t_npz)} — the exporter writes obs_spec.json last; re-run the "
                                          f"export (or let it finish)")
    try:
        with open(os.path.join(d, L.REQUIRED_FILES[0]), "rb") as f:
            raw = f.read()
        with open(os.path.join(d, L.REQUIRED_FILES[1]), "rb") as f:
            npz = f.read()
    except OSError as e:
        return skip(SKIP_MISSING, f"unreadable ({e.strerror or e})")
    hashes = {"obs_sha1": hashlib.sha1(raw).hexdigest(), "npz_sha1": hashlib.sha1(npz).hexdigest()}
    try:
        spec = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return skip(SKIP_JSON, str(e)[:160], **hashes)
    if not isinstance(spec, dict):
        return skip(SKIP_JSON, "not a JSON object", **hashes)
    it = spec.get("rsl_rl_iteration")
    if isinstance(it, bool) or not isinstance(it, int):
        return skip(SKIP_ITERATION, "no integer rsl_rl_iteration — cannot be told from a smoke export", **hashes)
    if it < min_iteration:
        return skip(SKIP_ITERATION, f"rsl_rl_iteration {it} < {min_iteration}", **hashes)
    reason, detail = check_weights(spec, npz)
    if reason:
        return skip(reason, detail, **hashes)
    e = _entry_for(d)
    if e is None or e.error or (e.obs_sha1, e.npz_sha1) != (hashes["obs_sha1"], hashes["npz_sha1"]):
        why = e.error if e is not None and e.error else "the files changed between two reads — sync again"
        return skip(SKIP_CHANGED, why, **hashes)
    return Item(d, src.tag, IMPORT, spec=spec, src=src, entry=e, **hashes)


# ----------------------------------------------------------------------------- naming
class Namespace:
    """The names the library may not reuse for other bytes: the station folder's (READ ONLY), the library's, and the
    ones this sync has already given out (a dry run writes nothing, so it must remember its own plan)."""

    def __init__(self, dst: str, station: str | None):
        self.dst, self.station = dst, station
        self.planned: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _hashes(p: str) -> tuple[str, str] | None:
        try:
            return L.export_hashes(p)
        except OSError:
            return None                                 # held, but not by a readable export: other bytes

    def holders(self, name: str) -> list[tuple[str, tuple | None]]:
        """[(who, (obs_sha1, npz_sha1) | None)] for everything holding `name` (any directory entry counts)."""
        out = []
        for root in (self.station, self.dst):
            if root:
                p = os.path.join(root, name)
                if os.path.lexists(p):
                    out.append((L.rel_to_repo(p), self._hashes(p)))
        if name in self.planned:
            out.append(("this sync", self.planned[name]))
        return out

    def family(self, tag: str) -> list[str]:
        """Every '<tag>@*' name already held — in the station folder, the library or this sync's plan — sorted."""
        pre = f"{tag}@"
        names = {n for n in self.planned if n.startswith(pre)}
        for root in (self.station, self.dst):
            if root:
                try:
                    names.update(n for n in os.listdir(root) if n.startswith(pre))
                except OSError:
                    pass                                # a root that does not exist (yet) holds nothing
        return sorted(names)

    def choose(self, tag: str, stamp: _dt.datetime | None, hashes: tuple[str, str]) -> tuple[str, str, str]:
        """(IMPORT | UP_TO_DATE | SKIP, name, who/why) for an export: the same bytes under ANY name of the tag's family
        (<tag>, <tag>@<stamp>, then every other '<tag>@*' held) = up to date; else the first of <tag> / <tag>@<stamp>
        nobody holds; else a skip. Never a name already held."""
        cands = [tag] + ([f"{tag}@{stamp:%Y%m%d-%H%M%S}"] if stamp is not None else [])
        # 2026-11-02 (review): the two candidates alone missed an earlier '@stamp' copy whenever the stamp moved — a
        # Policies/ stamp is obs_spec.json's mtime (a touch, a git checkout round trip, a cp without -p), so the same
        # bytes came in again as '<tag>@<new stamp>'. Looking up the whole family costs 0-2 hashes per source today.
        for c in cands + [n for n in self.family(tag) if n not in cands]:
            for who, h in self.holders(c):
                if h == tuple(hashes):
                    return UP_TO_DATE, c, who
        if tag == "exported":                           # rl_library.export_name reads '<x>/exported' as a raw run
            return SKIP, "", "'exported' cannot be a library name (rl_library.export_name reads it as a raw run)"
        for c in cands:
            if not self.holders(c):
                self.planned[c] = tuple(hashes)
                return IMPORT, c, ""
        held = "; ".join(f"{c}: {', '.join(w for w, _ in self.holders(c))}" for c in cands)
        return SKIP, "", f"held by other bytes ({held})"


def plan(rl_root: str, dst: str, station: str | None = None, min_iteration: int = L.MIN_ITERATION,
         settle_s: float = 120.0, now: float | None = None) -> list[Item]:
    """Every decision a sync would take, in source order (folder-level skips first). Reads only — a dry run is
    exactly this. ``station`` (normally rov_gui/control/rl_policies) joins the namespace; None leaves it out."""
    now = time.time() if now is None else float(now)
    sources, items = discover(rl_root)
    names = Namespace(dst, station)
    for src in sources:
        it = inspect_export(src, min_iteration, settle_s, now)
        if it.action == IMPORT:
            action, name, who = names.choose(src.tag, src.stamp, (it.obs_sha1, it.npz_sha1))
            it.action, it.name = action, name
            if action == UP_TO_DATE:
                it.held_by = who
            elif action == SKIP:
                it.reason, it.detail = SKIP_NAME, who
        items.append(it)
    return items


# ----------------------------------------------------------------------------- install
_TOP_KEY_RE = re.compile(r"^(hydro_model|cfd_drag_path):[ \t]*(.*?)[ \t]*$")


def read_hydro(run_dir: str | None) -> dict | None:
    """{hydro_model, cfd_drag_path} (those present) from <run>/params/env.yaml, else None. obs_spec.json cannot tell a
    CFD-drag run from a box-drag one (pwm11att14Rw_s1 vs pwm11att14Rcfd_s1 differ only in 'checkpoint'; RL_controller
    logs/rsl_rl/rov_posehold/2026-10-30_19-41-13_pwm11att14Rcfd_s1/params/env.yaml:420-421) — only env.yaml can.
    A LINE scan of two top-level keys, never a YAML load: env.yaml carries !!python/tuple tags that only an unsafe
    loader constructs. Provenance only; never fails a sync."""
    if not run_dir:
        return None
    out = {}
    try:
        with open(os.path.join(run_dir, "params", "env.yaml"), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _TOP_KEY_RE.match(line.rstrip("\n"))
                if m and m.group(1) not in out:
                    v = m.group(2)
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                        v = v[1:-1]
                    out[m.group(1)] = None if v in ("", "null", "~") else v
    except OSError:
        return None
    return out or None


def _copy_set(d: str) -> list[str]:
    """The files of an export that go into the library: REQUIRED_FILES, then OPTIONAL_FILES and README*.md present."""
    names = sorted(os.listdir(d))
    extra = [f for f in OPTIONAL_FILES if f in names] + [f for f in names if _README_RE.match(f)]
    return [f for f in list(L.REQUIRED_FILES) + extra if os.path.isfile(os.path.join(d, f))]


def _fsync(p: str) -> None:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sidecar(it: Item, files: dict, synced_at: str) -> dict:
    """LIBRARY.json for an import (rl_library.SIDECAR; schema 1). rl_library.scan() reads source_dir, run_stamp,
    rsl_rl_iteration and verifies obs_sha1 / npz_sha1 against the files on EVERY read (an edited entry becomes an
    error entry, never offered)."""
    src, e, spec = it.src, it.entry, it.spec
    return {
        "schema": SIDECAR_SCHEMA,
        "name": it.name,
        "tag": src.tag,
        "source_dir": src.path,
        "source_run": src.run,
        "run_stamp": src.stamp.isoformat() if src.stamp is not None else None,
        "stamp_from": src.stamp_from,
        "synced_at": synced_at,
        "files": files,
        "obs_sha1": it.obs_sha1,
        "npz_sha1": it.npz_sha1,
        "id": it.id,
        "rsl_rl_iteration": spec.get("rsl_rl_iteration"),
        "checkpoint": spec.get("checkpoint"),
        "dr_preset": spec.get("dr_preset"),
        "action_mode": e.action_mode,                   # as the station reads it: no key = "axes"
        "attitude_trained": e.att_trained,
        "cap_trained": e.cap_trained,                   # pwm exports: output.pwm_model.pwm_cap_trained, else None
        "hydro": read_hydro(src.run_dir),
    }


def install(it: Item, dst: str, synced_at: str | None = None) -> str:
    """Copy one planned import into <dst>/<name> (module docstring, AN IMPORT). The absolute entry folder; raises on
    any failure, with the temp folder removed and nothing else touched."""
    final = os.path.join(dst, it.name)
    if os.path.lexists(final):
        raise FileExistsError(f"{L.rel_to_repo(final)} exists — a library entry is never overwritten")
    synced_at = synced_at or _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    os.makedirs(dst, exist_ok=True)
    tmp = os.path.join(dst, f"{TMP_PREFIX}{it.name}-{os.getpid()}")
    os.mkdir(tmp)                                       # exists = another sync of this pid is mid-copy: not ours
    try:
        files = {}
        for f in _copy_set(it.source):
            shutil.copyfile(os.path.join(it.source, f), os.path.join(tmp, f))
            files[f] = L.file_sha1(os.path.join(tmp, f))
        got = (files.get(L.REQUIRED_FILES[0]), files.get(L.REQUIRED_FILES[1]))
        if got != (it.obs_sha1, it.npz_sha1):
            raise RuntimeError("the export changed while it was copied — not imported; sync again")
        with open(os.path.join(tmp, L.SIDECAR), "w", encoding="utf-8") as f:
            json.dump(sidecar(it, files, synced_at), f, indent=1)
            f.write("\n")
        for f in sorted(os.listdir(tmp)):
            p = os.path.join(tmp, f)
            os.chmod(p, 0o444)
            _fsync(p)                                   # a power cut after the rename must not leave empty files
        _fsync(tmp)
        if os.path.lexists(final):
            raise FileExistsError(f"{L.rel_to_repo(final)} appeared meanwhile — not overwritten")
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        _fsync(dst)                                     # the rename itself durable — best effort: it has happened
    except OSError:
        pass
    return final


def sync(rl_root: str, dst: str, station: str | None = None, min_iteration: int = L.MIN_ITERATION,
         settle_s: float = 120.0, dry_run: bool = False, now: float | None = None) -> list[Item]:
    """plan(), then (unless dry_run) install() every IMPORT; a failed one becomes FAILED with why."""
    items = plan(rl_root, dst, station, min_iteration, settle_s, now)
    if dry_run:
        return items
    synced_at = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    for it in items:
        if it.action != IMPORT:
            continue
        try:
            install(it, dst, synced_at)
        except Exception as e:                          # noqa: BLE001 — one bad export never stops the others
            it.action, it.detail = FAILED, f"{type(e).__name__}: {e}"
    return items


# ----------------------------------------------------------------------------- output
def _rel(p: str) -> str:
    return L.rel_to_repo(p)


def offered_to(e: L.Entry | None) -> list[str]:
    """The RL LOW keys whose picker can offer this export — rl_library.refusal_reason, the rule candidates() and the
    picker dialog use (so `list` can never call an export offered that the picker refuses) — [] when none."""
    if e is None:
        return []
    return [low for low in L.RL_LOWS if L.refusal_reason(e, low) is None]


def _lows_cell(e: L.Entry | None) -> str:
    if e is not None and e.no_fly is not None:
        return "NO-FLY"
    lows = offered_to(e)
    return ",".join(L.LOW_LABELS[x] for x in lows) if lows else "no LOW"


def _num(v, fmt="{:g}") -> str:
    return "—" if v is None else fmt.format(v)


def _table(rows: list[list[str]], indent: str = "  ") -> list[str]:
    """Left-aligned columns, the LAST one unpadded (it holds the long free text)."""
    if not rows:
        return []
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    return [(indent + "  ".join(c.ljust(w[i]) for i, c in enumerate(r[:-1])) + "  " + r[-1]).rstrip() for r in rows]


def _wrap(names: list[str], indent: str = "      ") -> list[str]:
    return textwrap.wrap(", ".join(names), width=116, initial_indent=indent, subsequent_indent=indent,
                         break_on_hyphens=False) or [indent]


def counts(items: list[Item]) -> tuple[dict, dict]:
    """({action: n}, {skip reason: n} in SKIP_ORDER, non-zero only) — the summary line's numbers."""
    n = {a: sum(1 for i in items if i.action == a) for a in (IMPORT, UP_TO_DATE, SKIP, FAILED)}
    by = {r: k for r in SKIP_ORDER for k in [sum(1 for i in items if i.action == SKIP and i.reason == r)] if k}
    return n, by


def print_sync(items: list[Item], rl_root: str, dst: str, station: str | None, dry_run: bool, min_iteration: int,
               settle_s: float, out=None) -> None:
    out = out or sys.stdout

    def p(s=""):
        print(s, file=out)

    verb = "would import" if dry_run else "imported"
    runs = _subdirs(os.path.join(rl_root, RUNS_SUBDIR))
    pols = _subdirs(os.path.join(rl_root, POLICIES_SUBDIR))
    p("RL policy library sync" + (" — DRY RUN: nothing is written" if dry_run else ""))
    p(f"  from     {rl_root}  ({RUNS_SUBDIR}/: {len(runs) if runs is not None else 'absent'} folders · "
      f"{POLICIES_SUBDIR}/: {len(pols) if pols is not None else 'absent'})")
    p(f"  into     {_rel(dst)}" + ("" if os.path.isdir(dst) else "  (absent — the first import creates it)"))
    if station:
        p(f"  station  {_rel(station)}  (read only; its names are never reused for other bytes)")
    p(f"  rules    rsl_rl_iteration >= {min_iteration}; no file newer than {settle_s:g} s")
    if min_iteration < L.MIN_ITERATION:
        p(f"  NOTE     --min-iteration {min_iteration} is below rl_library.MIN_ITERATION {L.MIN_ITERATION}: an export "
          f"in between is imported but the picker never offers it (smoke-length)")
    imports = [i for i in items if i.action == IMPORT]
    p()
    p(f"{verb} ({len(imports)}):")
    rows = [["NAME", "OFFERED TO", "CAP", "ITER", "TRAINED", "ID", "FROM"]]
    for i in imports:
        e = i.entry
        rows.append([i.name, _lows_cell(e), _num(e.cap_trained if e else None),
                     _num(i.spec.get("rsl_rl_iteration"), "{}"), (e.trained if e else "") or "—", i.id,
                     (i.src.run if i.src is not None and i.src.run else os.path.relpath(i.source, rl_root))])
    for line in _table(rows if imports else []):
        p(line)
    fails = [i for i in items if i.action == FAILED]
    if fails:
        p()
        p(f"FAILED ({len(fails)}) — nothing of these was written:")
        for i in fails:
            p(f"  {i.name or i.tag}: {i.detail}  ({os.path.relpath(i.source, rl_root)})")
    ups = [i for i in items if i.action == UP_TO_DATE]
    if ups:
        p()
        p(f"up to date ({len(ups)}):")
        groups: dict[str, list[str]] = {}
        for i in ups:
            groups.setdefault(os.path.dirname(i.held_by) if i.held_by != "this sync" else "this sync",
                              []).append(i.name)
        for where, names in groups.items():
            p(f"    same bytes in {where} ({len(names)}):")
            for line in _wrap(names):
                p(line)
    skips = [i for i in items if i.action == SKIP]
    if skips:
        p()
        p(f"skipped ({len(skips)}):")
        for r in SKIP_ORDER:
            grp = [i for i in skips if i.reason == r]
            if not grp:
                continue
            p(f"    {r} ({len(grp)}):")
            with_detail = r not in (SKIP_NOT_RUN, SKIP_NO_STAMP, SKIP_NO_TAG, SKIP_SMOKE, SKIP_NO_EXPORT)
            for line in _wrap([f"{i.tag} ({i.detail})" if with_detail and i.detail else i.tag for i in grp]):
                p(line)
    tmp = [n for n in (sorted(os.listdir(dst)) if os.path.isdir(dst) else []) if n.startswith(TMP_PREFIX)]
    if tmp:
        p()
        p(f"stale temp folders in {_rel(dst)} (an interrupted sync; never listed — delete by hand): {', '.join(tmp)}")
    n, by = counts(items)
    p()
    p(f"summary: {verb} {n[IMPORT]} · up to date {n[UP_TO_DATE]} · skipped {n[SKIP]}"
      + (f" ({' · '.join(f'{r} {k}' for r, k in by.items())})" if by else "")
      + (f" · FAILED {n[FAILED]}" if n[FAILED] else ""))
    if imports:
        per = " · ".join(f"{L.LOW_LABELS[low]} {sum(1 for i in imports if low in offered_to(i.entry))}"
                         for low in L.RL_LOWS)
        nofly = sum(1 for i in imports if i.entry is not None and i.entry.no_fly is not None)
        nolow = sum(1 for i in imports if (i.entry is None or i.entry.no_fly is None) and not offered_to(i.entry))
        p(f"  of those, offered to {per}; no-fly {nofly} (listed greyed, refused at the pick); no LOW flies {nolow}")
    if dry_run:
        p("  nothing written — run without --dry-run to import")
    elif n[IMPORT]:
        p("  the picker lists them the next time it opens (it rescans; no station restart)")


def _sort_newest(entries: list) -> list:
    out = sorted(entries, key=lambda e: e.name)
    out.sort(key=lambda e: e.trained, reverse=True)     # = rl_library.candidates' order
    return out


def print_list(station: str, dst: str, low: str | None = None, include_refused: bool = False, out=None) -> None:
    """The station + library exports as the picker sees them (rl_library.dedupe(scan())): every one some RL LOW is
    offered — or, with ``low``, rl_library.candidates for that LOW (the picker's own list, newest first); with
    ``include_refused`` the refused ones too, each with why."""
    out = out or sys.stdout

    def p(s=""):
        print(s, file=out)

    raw = L.scan([("station", station), ("library", dst)])
    ents = L.dedupe(raw)
    n_st = sum(1 for e in raw if e.where == "station")
    n_lib = sum(1 for e in raw if e.where == "library")
    p(f"RL policy library — station {_rel(station)} ({n_st}) · library {_rel(dst)} ({n_lib}"
      + ("" if os.path.isdir(dst) else ", absent — ./c3 rl-policies sync fills it") + f") · {len(ents)} after dedupe")
    if low is not None:
        rows_e = L.candidates(ents, low, include_refused=include_refused)
        n_ok = len(L.candidates(ents, low))
        p(f"{L.LOW_LABELS[low]} ({low}): {n_ok} offered by the LOW row's picker, newest first"
          + (f" + {len(rows_e) - n_ok} refused (the picker greys them)" if include_refused else ""))
    else:
        rows_e = _sort_newest([e for e in ents if include_refused or offered_to(e)])
        p(f"{sum(1 for e in ents if offered_to(e))} offered to some RL LOW, newest first"
          + ("; the rest with why" if include_refused else ""))
    rows = [["NAME", "WHERE", "TRAINED", "FITS", "CAP", "PRESET", "ITER", "VEL", "ID", "NOTE"]]
    for e in rows_e:
        if e.error:
            note = f"ERROR: {e.error}"
        elif e.no_fly is not None:
            note = f"NO-FLY: {e.no_fly}"
        elif low is not None and L.refusal_reason(e, low) is not None:
            note = f"not for {L.LOW_LABELS[low]}: {L.refusal_reason(e, low)}"
        elif not offered_to(e):                         # a smoke-length one: the floor's text, not the slots'
            note = "no LOW: " + "; ".join(sorted({L.refusal_reason(e, x) for x in L.RL_LOWS} - {None}))
        else:
            note = ""
        fits = [L.LOW_LABELS[x] for x in L.RL_LOWS if e.fit.get(x, "not evaluated") is None]
        rows.append([e.name, e.where, e.trained or "—", ",".join(fits) or "—", _num(e.cap_trained), e.dr_preset or "—",
                     _num(e.iteration, "{}"), "DVL blend" if e.vel_trained else "tag fd", e.id or "—", note])
    for line in _table(rows if rows_e else []):
        p(line)
    hidden = len(ents) - len(rows_e)
    if hidden:
        why = f"not for {L.LOW_LABELS[low]}" if low is not None else "no fitting LOW"
        p(f"({hidden} more not listed — no-fly, {why}, smoke-length or unreadable; --all lists them with why)")
    # one name, two different exports (the station's copy and another): legal, but the operator must pick by id
    ids: dict[str, set] = {}
    for e in raw:
        if e.id and not e.error:
            ids.setdefault(e.name, set()).add((e.where, e.id))
    for name, held in sorted(ids.items()):
        if len({i for _, i in held}) > 1:
            p(f"NOTE: '{name}' names {len({i for _, i in held})} different exports ("
              + ", ".join(f"{w} id {i}" for w, i in sorted(held)) + ") — tell them apart by id")


# ----------------------------------------------------------------------------- CLI
def _low_arg(v: str) -> str:
    k = str(v).strip().lower()                          # the label (RL_PWM_7D) lowercased IS the key
    if k not in L.RL_LOWS:
        raise argparse.ArgumentTypeError(f"{v!r} is not an RL LOW (one of {', '.join(L.RL_LOWS)})")
    return k


def _inside(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip(os.sep) + os.sep)


def dst_refusal(dst: str, station: str, rl_root: str) -> str | None:
    """Why `sync` must not write into ``dst`` (all three absolute, rl_library.resolve_dir form), or None: (inside) the
    station folder, inside RL_controller, or inside the repo anywhere but its git-ignored data/. Checked on the paths
    as given (these texts unchanged), then AGAIN on their realpaths — 2026-11-02 (review): resolve_dir is lexical by
    design (what gets recorded), so a --dst that is or sits under a symlink into the station folder or RL_controller,
    an --rl-root given through an alias, or a data/ symlinked into RL_controller (to "share" Policies/) passed every
    text check, and install() would have written ~170 folders through the link. The repo's data/ is realpath'd as the
    ALLOWED subtree too, so a data/ that is a symlink to another disk is still allowed."""
    repo = str(L.REPO_ROOT)
    lexical = (dst, station, rl_root, repo, os.path.join(repo, "data"))
    physical = tuple(os.path.realpath(p) for p in lexical)        # non-strict: a dst not created yet still resolves
    for phys, (d, st, rl, rp, da) in ((False, lexical), (True, physical)):
        if _inside(d, st):
            how = f" through a symlink ({d} is inside {st})" if phys else ""
            return (f"rl-policies sync: --dst {dst} is (inside) the station folder {station}{how} — that one is "
                    f"git-tracked and curated by hand; the library is {L.LIBRARY_ROOTS[1][1]}")
        if _inside(d, rl):
            how = f" through a symlink ({d} is inside {rl})" if phys else ""
            return f"rl-policies sync: --dst {dst} is inside RL_controller ({rl_root}){how}, which is read only"
        if _inside(d, rp) and not _inside(d, da):
            # inside the repo only under data/ (.gitignore: data/): elsewhere ~170 export folders would land among
            # tracked sources (a mistyped --dst rov_gui/control would put them beside rl_policy.py)
            how = f" through a symlink ({d} is inside {rp}, not under {da})" if phys else ""
            return (f"rl-policies sync: --dst {dst} is inside the repo{how} but not under its git-ignored data/ — "
                    f"the library is {L.LIBRARY_ROOTS[1][1]}")
    return None


def cmd_sync(a) -> int:
    rl_root = L.resolve_dir(a.rl_root) if a.rl_root else default_rl_root()
    dst, station = L.resolve_dir(a.dst), L.resolve_dir(a.station)
    why = dst_refusal(dst, station, rl_root)
    if why:
        raise SystemExit(why)
    if a.min_iteration < 0 or not a.settle_s >= 0.0:
        raise SystemExit("rl-policies sync: --min-iteration and --settle-s must be >= 0")
    if not any(os.path.isdir(os.path.join(rl_root, s)) for s in (RUNS_SUBDIR, POLICIES_SUBDIR)):
        raise SystemExit(f"rl-policies sync: {rl_root} has neither {RUNS_SUBDIR}/ nor {POLICIES_SUBDIR}/ — is "
                         f"--rl-root (or ${RL_ROOT_ENV}) right?")
    items = sync(rl_root, dst, station, a.min_iteration, a.settle_s, dry_run=a.dry_run)
    print_sync(items, rl_root, dst, station, a.dry_run, a.min_iteration, a.settle_s)
    return 1 if any(i.action == FAILED for i in items) else 0


def cmd_list(a) -> int:
    print_list(L.resolve_dir(a.station), L.resolve_dir(a.dst), a.low, a.all)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def roots(s):
        s.add_argument("--dst", default=L.LIBRARY_ROOTS[1][1], metavar="DIR",
                       help="the library folder (default %(default)s; relative = repo-root-relative)")
        s.add_argument("--station", default=L.LIBRARY_ROOTS[0][1], metavar="DIR",
                       help="the git-tracked station export folder — only ever READ (default %(default)s)")

    s = sub.add_parser("sync", help="copy new RL_controller exports into the library (never overwrites, never deletes)")
    s.add_argument("--rl-root", default=None, metavar="DIR",
                   help=f"the RL_controller checkout, read only (default ${RL_ROOT_ENV}, else {DEFAULT_RL_ROOT})")
    roots(s)
    s.add_argument("--dry-run", action="store_true", help="say what would happen; write nothing")
    s.add_argument("--min-iteration", type=int, default=L.MIN_ITERATION, metavar="N",
                   help="skip exports with rsl_rl_iteration below N (smoke runs export at 1-2; default %(default)s = "
                        "rl_library.MIN_ITERATION, below which the picker never offers an export anyway)")
    s.add_argument("--settle-s", type=float, default=120.0, metavar="S",
                   help="skip an export any of whose files changed less than S seconds ago (default %(default)g)")
    s = sub.add_parser("list", help="what the LOW row's picker can offer (station + library, deduplicated by id)")
    s.add_argument("--low", type=_low_arg, default=None, metavar="LOW",
                   help="rl | rl_pwm | rl_7d | rl_pwm_7d (or its label): exactly that LOW's picker list")
    s.add_argument("--all", action="store_true",
                   help="also the refused ones (no-fly, no fitting LOW, smoke-length, unreadable), with why")
    roots(s)
    a = p.parse_args(argv)
    return {"sync": cmd_sync, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
