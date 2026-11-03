#!/usr/bin/env python3
"""rl_library.py — the RL POLICY LIBRARY: every RL export the station can fly, listed from two folders, identified by
content hash and pre-sorted per RL LOW for the LOW row's picker (2026-11-01, operator request: "rl_7d_pwm 여러 개를
훈련해 왔다. GUI 켜기 전에 hw_mpc.yaml 에서 policy_dir 을 정하는 대신 GUI 에서 바로 바꾸고 싶다. 그리고 RL 훈련한
것들을 어떤 폴더에 다 모아두고.").

An EXPORT is a directory holding the two files the station reads (REQUIRED_FILES: obs_spec.json + policy_weights.npz —
HwRl reads nothing else; policy.onnx / policy_wrench.onnx / READMEs may ride along and are never opened). Two roots,
listed in priority order (LIBRARY_ROOTS):

  station   rov_gui/control/rl_policies/   git-tracked; the exports hw_mpc.yaml names and that tools/tests pin
                                           (install_rl_pwm_lua.py and test_rl_pwm.py read pwm10_s1) — kept exactly
                                           as it is
  library   data/rl_policies/              git-ignored (.gitignore: data/); filled by `./c3 rl-policies sync`
                                           (rov_gui/tools/rl_policy_library.py) from RL_controller
                                           logs/rsl_rl/rov_posehold/<YYYY-MM-DD_HH-MM-SS>_<tag>/exported/

Operator decision 2026-11-01: the library lives in data/rl_policies BESIDE the git-tracked station folder (every
trained export is too many files for git), and the picker lists BOTH. An export present in both (same id) is listed
once, as the station copy (dedupe).

IDENTITY is content, never the folder name: id = sha1(f"{sha1(obs_spec.json)}:{sha1(policy_weights.npz)}")[:12]
(export_id). Raw RL_controller export folders change in place (RL_controller logs/rsl_rl/rov_posehold/
2026-09-22_04-15-41_traj10_s1/exported/obs_spec.json was rewritten 2026-09-27 11:56 — sha1 efba0d34fcff vs the
station copy's da3759dc8e1d, hashed 2026-11-01; play.py overwrites policy.onnx; a same-tag rerun reuses the name),
and a run meta that records only policy_dir + checkpoint cannot tell two exports under one name apart — so the
worker records both sha1s and the id of whatever RL export it flies (meta controller rl_policy).

IMMUTABILITY: a library entry is written once (copied to a temp dir, then an atomic rename) and NEVER overwritten or
modified; a different export arriving under a name already used gets '<name>@<YYYYMMDD-HHMMSS>' (its run stamp), so a
name that a past run meta recorded keeps meaning the same bytes. The LIBRARY.json sidecar (SIDECAR) records where the
copy came from; when it carries obs_sha1 / npz_sha1, scan() turns an entry whose files no longer match them into an
error entry (never pickable) — an in-place edit is caught, not flown.

FLIGHT BAN (NO_FLY: pwm10_s1, pwm10_s2, att7pwm_s0/s1/s2) — by name (also '<name>@<stamp>'), by the weights' sha1 (a
renamed copy is still caught) and by the training run its obs_spec checkpoint names (no_fly_reason).

OFFERED (refusal_reason, 2026-11-02 — one rule for candidates(), the picker dialog and `./c3 rl-policies list`):
readable, not banned, at least MIN_ITERATION training iterations, and slot_fit's verdict for the LOW.

slot_fit(spec) is a CHEAP PRE-FILTER per RL LOW from obs_spec.json alone (action mode, output kind, the attitude
reference, the observation guard, the integral observation, ...). It never decides: every pick re-runs the follower's
own constructor (HwRl / HwRl7d / HwRlPwm / HwRlPwm7d — all build-time gates, including the config-dependent ones this
module cannot see: the wire cap vs the trained cap, the DVL blend vs velocity_source, rl_pwm.obs_guard, ctrl_hz), and
only that build's verdict flies — the follower build stays AUTHORITATIVE. The predicates are pinned so that a predicate
ACCEPT never meets an export-intrinsic build refusal on the real exports (rov_gui/tests/test_rl_library.py; the
2026-11-01 agreement check over the station folder and every RL_controller export).

No Qt, no numpy: json / hashlib / os / pathlib / dataclasses / datetime / re only — the GUI, the worker and the
`./c3 rl-policies` CLI import it fast.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: The repo root — the SAME value as rl_policy.REPO_ROOT (abspath, no symlink resolution: a relative policy_dir means
#: the same folder to the picker, the worker and the follower). Not imported from rl_policy: that pulls numpy + pid.
REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ----------------------------------------------------------------------------- shared names (DESIGN 2026-11-01 §0)
#: The four RL LOWs a pick can swap (MpcWorker.set_rl_policy): per-LOW followers HwRl / HwRlPwm / HwRl7d / HwRlPwm7d.
RL_LOWS = ("rl", "rl_pwm", "rl_7d", "rl_pwm_7d")
#: = rov_gui.widgets.trajectory.MODE_LABELS for these keys (pinned by test_rl_library.py) — not imported from there,
#: which would pull Qt into the worker and the CLI.
LOW_LABELS = {"rl": "RL", "rl_pwm": "RL_PWM", "rl_7d": "RL_7D", "rl_pwm_7d": "RL_PWM_7D"}
#: (where, repo-relative root), in PRIORITY order: dedupe() keeps the first copy of an id, so the station's.
LIBRARY_ROOTS = (("station", "rov_gui/control/rl_policies"), ("library", "data/rl_policies"))
#: The only two files the station reads from an export (HwRl.__init__: obs_spec.json, then NumpyPolicy on the npz).
REQUIRED_FILES = ("obs_spec.json", "policy_weights.npz")
#: The sync tool's provenance sidecar inside a library entry (rov_gui/tools/rl_policy_library.py writes it).
SIDECAR = "LIBRARY.json"
#: The fewest training iterations an export may have to be OFFERED (candidates) or imported (the sync tool's
#: --min-iteration default). RL_controller's SMOKE runs export after 1-2 iterations and their obs_spec is complete, so
#: they pass slot_fit and the follower build (2026-11-01 agreement check: smoke_att7_axes6, smoke_pwm11,
#: smoke_pwm11att, smoke_r2smokeT; rsl_rl_iteration 1-2) — this floor, not the build, keeps a hand-copied one out of
#: the picker. The lowest non-smoke export is 599 [측정: RL_controller logs/rsl_rl/rov_posehold/*/exported/obs_spec.json
#: rsl_rl_iteration, read 2026-11-02; 2026-09-16_14-04-57_small and its siblings].
MIN_ITERATION = 50

_BAN_PWM10 = ("flight ban — the pwm pool-failure policy (RL_controller README §13.8 'Never fly again, at any cap'; "
              "config/hw_mpc.yaml rl_pwm comment)")
_BAN_ATT7PWM = ("flight ban — two pool runs failed (co-contraction 26–44 A, sink) (RL_controller README §13.8; "
                "config/hw_mpc.yaml rl_pwm_7d comment)")
#: RL_controller README.md:713 (§13.8): "Never fly again, at any cap: pwm10_s1, pwm10_s2, att7pwm_s*". Refused at the
#: pick (MpcWorker.set_rl_policy) and shown, greyed, in the picker. pwm10_s1 / att7pwm_s0 stay ON DISK in the station
#: folder (install_rl_pwm_lua.py and the tests read them) — banned from flight, not from the repo.
NO_FLY = {"pwm10_s1": _BAN_PWM10, "pwm10_s2": _BAN_PWM10,
          "att7pwm_s0": _BAN_ATT7PWM, "att7pwm_s1": _BAN_ATT7PWM, "att7pwm_s2": _BAN_ATT7PWM}
#: The five banned exports' policy_weights.npz sha1, pinned 2026-11-01 from the RL_controller originals
#: (logs/rsl_rl/rov_posehold/2026-09-27_11-06-05_pwm10_s{1,2}/exported/ and 2026-10-03_15-44-54_att7pwm_s{0,1,2}/
#: exported/; the station copies rl_policies/pwm10_s1 and att7pwm_s0 hash the same). The npz is byte-deterministic
#: (np.savez, zip entries dated 1980-01-01), so a renamed copy OR a re-export of the same weights carries one of these.
#: no_fly_reason ALSO hashes the station copies at first use (DESIGN §0) — this table adds the three that have no
#: station copy (pwm10_s2, att7pwm_s1, att7pwm_s2).
_NO_FLY_NPZ_PINNED = {"0dcde6f712bc208c310ab4e38f385f0d7d80ebf3": "pwm10_s1",
                      "575816919d8417fcf5a33394538b392aa45f07b8": "pwm10_s2",
                      "01911a4b9a038f535e25bfb55091dc85bb95d486": "att7pwm_s0",
                      "04263991bdb6c3b9f7feea0f416e54ef1271ed82": "att7pwm_s1",
                      "6414dd360a51ff99f90dd5264a9cabde2a40fd86": "att7pwm_s2"}
#: the station copies whose npz sha1 no_fly_reason hashes lazily, once (DESIGN §0)
_NO_FLY_STATION_COPIES = ("pwm10_s1", "att7pwm_s0")
_banned_npz_cache: dict | None = None

#: An RL_controller run folder: '<YYYY-MM-DD_HH-MM-SS>' + optional '_<tag>' (rsl_rl's log dir naming).
_RUN_DIR_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})(?:_(.*))?$")
_MODEL_RE = re.compile(r"model_(\d+)\.pt$")


# ----------------------------------------------------------------------------- paths and hashes
def file_sha1(path) -> str:
    """sha1 hex of a file's bytes (read in 1 MiB chunks). OSError when it cannot be read."""
    h = hashlib.sha1()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_dir(path) -> str:
    """The absolute directory a policy_dir string means: '~' expanded, a RELATIVE path joined to REPO_ROOT (as HwRl
    does — never the cwd, which differs between `./c3`, pytest and a shell), then abspath (normalised, '..' folded).
    NOT realpath: a symlinked folder keeps its lexical path, the one the follower's meta records."""
    p = os.path.expanduser(str(path))
    if not os.path.isabs(p):
        p = os.path.join(str(REPO_ROOT), p)
    return os.path.abspath(p)


def rel_to_repo(path) -> str:
    """Repo-relative when the (resolved) path lies inside REPO_ROOT ('.' for the root itself), else ABSOLUTE — never a
    '../..' climb (os.path.relpath's answer for RL_controller paths, which reads as if it were in the repo)."""
    a = resolve_dir(path)
    root = str(REPO_ROOT)
    if a == root:
        return "."
    if a.startswith(root.rstrip(os.sep) + os.sep):
        return a[len(root.rstrip(os.sep)) + 1:]
    return a


def export_hashes(export_dir) -> tuple[str, str]:
    """(sha1 of obs_spec.json, sha1 of policy_weights.npz) of an export dir (resolved like resolve_dir). OSError
    (FileNotFoundError) when either is missing or unreadable — the caller decides what that means."""
    d = resolve_dir(export_dir)
    return file_sha1(os.path.join(d, REQUIRED_FILES[0])), file_sha1(os.path.join(d, REQUIRED_FILES[1]))


def export_id(obs_sha1: str, npz_sha1: str) -> str:
    """The export's content id: sha1(f"{obs_sha1}:{npz_sha1}") hex, first 12 — what the panel, the picker and the run
    meta show. Both halves count: a re-export can keep the npz bytes and change obs_spec — pwm12PI_s0: station copy
    obs_spec 53d86e89a658 vs RL_controller logs/rsl_rl/rov_posehold/2026-10-18_05-01-44_pwm12PI_s0/exported
    256cf8537674, npz 6af40d0b6915 in both (sha1 prefixes, hashed 2026-11-01) — and the obs_spec is what the station
    builds the frame from."""
    return hashlib.sha1(f"{obs_sha1}:{npz_sha1}".encode("utf-8")).hexdigest()[:12]


def _run_tag(run_dir_name: str) -> str | None:
    """'<YYYY-MM-DD_HH-MM-SS>_<tag>' -> tag; None when the name has no run stamp or no tag."""
    m = _RUN_DIR_RE.match(str(run_dir_name))
    return m.group(7) if m and m.group(7) else None


def _run_stamp(run_dir_name: str) -> _dt.datetime | None:
    """The datetime of an RL_controller run folder's stamp, None when the name has none (or it is not a real date)."""
    m = _RUN_DIR_RE.match(str(run_dir_name))
    if not m:
        return None
    try:
        return _dt.datetime(*(int(m.group(i)) for i in range(1, 7)))
    except ValueError:
        return None


def export_name(export_dir) -> str:
    """The export's NAME: the folder's basename — except a raw RL_controller '<YYYY-MM-DD_HH-MM-SS>_<tag>/exported',
    which is named by its run tag (the station and the library name an export after its run's tag; the rsl_rl log dir
    is that tag behind the run stamp). A run folder without a tag keeps its whole name."""
    d = resolve_dir(export_dir)
    base = os.path.basename(d)
    if base == "exported":
        run = os.path.basename(os.path.dirname(d))
        return _run_tag(run) or run
    return base


def is_export_dir(path) -> bool:
    """A directory (resolved like resolve_dir) holding both REQUIRED_FILES as files. Never raises."""
    try:
        d = resolve_dir(path)
        return os.path.isdir(d) and all(os.path.isfile(os.path.join(d, f)) for f in REQUIRED_FILES)
    except (OSError, ValueError, TypeError):
        return False


# ----------------------------------------------------------------------------- flight ban
def _banned_npz_sha1() -> dict:
    """{npz sha1: banned name}: the pinned table + the station copies' npz, hashed lazily ONCE (a missing copy is
    skipped — the pinned table still covers it)."""
    global _banned_npz_cache
    if _banned_npz_cache is None:
        d = dict(_NO_FLY_NPZ_PINNED)
        for name in _NO_FLY_STATION_COPIES:
            p = os.path.join(str(REPO_ROOT), LIBRARY_ROOTS[0][1], name, REQUIRED_FILES[1])
            try:
                d[file_sha1(p)] = name
            except OSError:
                pass
        _banned_npz_cache = d
    return _banned_npz_cache


def checkpoint_run(checkpoint) -> tuple[str | None, _dt.datetime | None]:
    """(run tag, run stamp) of the training run an obs_spec ``checkpoint`` path names — the last path component that
    is an RL_controller run folder ('logs/rsl_rl/rov_posehold/2026-10-19_03-25-36_pwm16N_s3/model_2399.pt' ->
    ('pwm16N_s3', 2026-10-19 03:25:36)); (None, None) when there is none."""
    parts = [p for p in re.split(r"[\\/]+", str(checkpoint or "")) if p]
    for p in reversed(parts):
        st = _run_stamp(p)
        if st is not None:
            return _run_tag(p), st
    return None, None


def no_fly_reason(name, npz_sha1=None, checkpoint=None) -> str | None:
    """Why this export must never fly, or None. Matched by NAME (a '<tag>@<stamp>' name by the part before '@'; a path
    by its export_name), by the weights' sha1 against the banned exports' (a RENAMED copy is still caught), and — when
    the caller passes the obs_spec ``checkpoint`` — by the training run it names (another copy or a re-export of a
    banned run). The text is NO_FLY's, plus how a copy was recognised."""
    n = str(name or "")
    if os.sep in n or "/" in n:
        n = export_name(n)
    base = n.split("@", 1)[0]
    if base in NO_FLY:
        return NO_FLY[base]
    if npz_sha1:
        hit = _banned_npz_sha1().get(str(npz_sha1).strip().lower())
        if hit:
            return f"{NO_FLY[hit]} — same weights as {hit} (policy_weights.npz sha1 {str(npz_sha1)[:12]})"
    if checkpoint:
        tag, _ = checkpoint_run(checkpoint)
        if tag in NO_FLY:
            return f"{NO_FLY[tag]} — trained as {tag} (obs_spec checkpoint {checkpoint})"
    return None


# ----------------------------------------------------------------------------- which LOW can fly it (§1a)
#: which LOWs fly which obs_spec action_mode (rl_policy: HwRl.ACCEPTED_MODES ("axes",), HwRl7d ("axes6",), HwRlPwm
#: ("pwm",), HwRlPwm7d = pwm + attitude_reference.trained); any other mode (wrench6, ...) is flown by none
_MODE_LOWS = {"axes": ("rl",), "axes6": ("rl_7d",), "pwm": ("rl_pwm", "rl_pwm_7d")}
_LEVEL = "level-trained — no roll/pitch reference"


def _blk(spec: dict, key: str):
    """(present, the block) — the block is None when present but not a dict."""
    if key not in spec:
        return False, {}
    v = spec[key]
    return True, (v if isinstance(v, dict) else None)


def _flag(spec: dict, key: str, sub: str) -> tuple[bool, str]:
    """A ``{<sub>: true|false, ...}`` block as rl_policy reads it (check_obs_guard / check_velocity_source /
    obs_integral_enabled): (value, "") — absent = False — or (False, why) for a block every LOW refuses."""
    present, b = _blk(spec, key)
    if not present:
        return False, ""
    if b is None or not isinstance(b.get(sub), bool):
        return False, f"malformed {key} block (needs {sub}: true|false) — refused by every LOW"
    return b[sub], ""


def _pos_finite(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return not isinstance(v, bool) and math.isfinite(f) and f > 0.0


def slot_fit(spec: dict) -> dict:
    """{low: None (fits) | a short reason} for each of RL_LOWS — a CHEAP pre-filter from obs_spec.json alone (DESIGN
    §1a). Never raises. The follower build stays AUTHORITATIVE: a pick re-runs the LOW's constructor, which also
    applies the config-dependent gates this cannot see (cap vs trained cap, velocity_source vs the DVL blend,
    rl_pwm.obs_guard, ctrl_hz, the sign probe). What is checked here is only what the build refuses whatever the
    config ("export-intrinsic"), so a fit here never meets such a refusal on the real exports (2026-11-01 agreement
    check; rov_gui/tests/test_rl_library.py):

      every LOW  the blocks rl_policy reads as {flag: bool} (obs_guard.trained, velocity_source.trained,
                 obs_integral.enabled) well-formed, obs_goal_frame a bool, history_len / frame_dim / num_actions
                 integers (HwRl's int(spec[...])); action_mode one the station flies (axes / axes6 / pwm)
      rl         axes; not trained behind the pwm obs guard (check_obs_guard(None)), no integral observation
                 (OBS_INTEGRAL_LOW False)
      rl_7d      axes6 + attitude_reference.trained (HwRl7d pre-check); the same guard / integral refusals; the 7D
                 layout pins the goal feed-forward and 6 outputs (check_frame_layout(gv=True, na=6)); the trained
                 envelope roll_max_rad / pitch_max_rad / rate_max_rad_s finite > 0
      rl_pwm     pwm with output.kind pwm and an output.pwm_model object; 8 outputs (HwRlPwm)
      rl_pwm_7d  rl_pwm's + attitude_reference.trained (HwRlPwm7d pre-check); feed-forward + 8 outputs pinned; the
                 envelope as rl_7d
    rl_pwm also flies attitude-trained pwm exports — LEVEL (q_des without roll/pitch; HwRl._q_des): it is the build's
    behaviour, and the picker says which LOW tracks roll/pitch by its slot.
    Deliberately NOT transcribed (export-intrinsic, but they need rl_policy's own tables or the npz, and a copy here
    would silently HIDE exports once rl_policy learns a new layout or law): the frame_layout / frame_dim match
    (check_frame_layout), the integral law's keys and texts (check_obs_integral), the PwmModel internals, the npz
    dimensions. No real export trips them (2026-11-01); a pick of one that does is refused by the build, with its
    reason — visible, never flown."""
    if not isinstance(spec, dict):
        return {low: "obs_spec.json is not a JSON object" for low in RL_LOWS}
    mode = str(spec.get("action_mode", "axes"))
    # ---- refused by every LOW (HwRl.__init__ before any per-LOW gate) ----
    common = ""
    og, why_og = _flag(spec, "obs_guard", "trained")
    integral, why_int = _flag(spec, "obs_integral", "enabled")
    _vs, why_vs = _flag(spec, "velocity_source", "trained")
    for why in (why_og, why_int, why_vs):
        common = common or why
    if not common and not isinstance(spec.get("obs_goal_frame", False), bool):
        common = "obs_goal_frame is not true/false — refused by every LOW"
    if not common:
        for k in ("history_len", "frame_dim", "num_actions"):
            try:
                int(spec[k])
            except (KeyError, TypeError, ValueError):
                common = f"obs_spec has no integer {k} — refused by every LOW"
                break
    if not common and mode not in _MODE_LOWS:
        common = f"action_mode {mode} is not flown by the station"
    if common:
        return {low: common for low in RL_LOWS}
    na = int(spec["num_actions"])
    _, ar = _blk(spec, "attitude_reference")
    att = bool(ar.get("trained")) if isinstance(ar, dict) else False
    out = spec.get("output")
    out = out if isinstance(out, dict) else {}
    fit = {}
    for low in RL_LOWS:
        lows = _MODE_LOWS[mode]
        if low not in lows:
            fit[low] = f"{mode} export — {' / '.join(LOW_LABELS[x] for x in lows)} only"
            continue
        why = None
        if low in ("rl", "rl_7d"):
            if og:
                why = "trained with the pwm obs guard — pwm LOWs only"
            elif integral:
                why = "integral observation — pwm LOWs only"
        if why is None and low in ("rl_pwm", "rl_pwm_7d"):
            if str(out.get("kind")) != "pwm" or not isinstance(out.get("pwm_model"), dict):
                why = "no output.pwm_model (kind pwm) — re-export"
            elif na != 8:
                why = f"{na} outputs — the pwm transport carries 8"
        if why is None and low in ("rl_7d", "rl_pwm_7d"):
            if ar is None:
                why = "attitude_reference is not an object"
            elif not att:
                why = _LEVEL
            elif not bool(spec.get("obs_goal_velocity", False)):
                why = "no goal feed-forward — the 7D layout needs it"
            elif low == "rl_7d" and na != 6:
                why = f"{na} outputs — RL_7D flies 6"
            elif not all(_pos_finite(ar.get(k)) for k in ("roll_max_rad", "pitch_max_rad", "rate_max_rad_s")):
                why = "attitude_reference envelope missing (roll/pitch/rate max)"
        fit[low] = why
    return fit


# ----------------------------------------------------------------------------- entries
@dataclass(frozen=True)
class Entry:
    """One export as the picker lists it. Built by scan(); ``fit`` is a plain dict (left out of ==/hash)."""

    name: str                           # export_name: the folder name (library: '<tag>' or '<tag>@<YYYYMMDD-HHMMSS>')
    path: str                           # absolute export dir (resolve_dir: lexical, no realpath)
    rel: str                            # rel_to_repo(path)
    where: str                          # "station" | "library" | "other"
    obs_sha1: str = ""
    npz_sha1: str = ""
    id: str = ""                        # export_id(obs_sha1, npz_sha1)
    action_mode: str = ""               # obs_spec action_mode (absent = "axes", as HwRl reads it)
    att_trained: bool = False           # attitude_reference.trained
    output_kind: str = ""               # output.kind ("pwm" | "wrench" | ...)
    cap_trained: float | None = None    # pwm exports: output.pwm_model.pwm_cap_trained (rl_pwm.PwmModel; absent = 1.0)
    dr_preset: str = ""
    iteration: int | None = None        # rsl_rl_iteration (else the sidecar's, else the checkpoint's model_N)
    checkpoint: str = ""                # obs_spec checkpoint (the training run's model path)
    trained: str = ""                   # "YYYY-MM-DD HH:MM" from the checkpoint's run stamp; else sidecar run_stamp; else ""
    vel_trained: bool = False           # velocity_source.trained (DVL-primary blend)
    integral: bool = False              # obs_integral.enabled
    fit: dict = field(default_factory=dict, compare=False)   # slot_fit(spec); a missing key counts as refused
    no_fly: str | None = None           # no_fly_reason(name, npz_sha1, checkpoint)
    source: str = ""                    # sidecar "source_dir" when present, else ""
    error: str = ""                     # "" or why the entry cannot be offered (other fields defaulted where unknown)


#: scan()'s per-path cache: abspath -> (key, Entry); key = where + (mtime_ns, size, inode) of obs_spec.json,
#: policy_weights.npz and the sidecar, taken BEFORE the files are read (a change during the read re-reads next scan)
_CACHE: dict = {}


def _stat_key(p: str):
    try:
        st = os.stat(p)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _fmt_stamp(t: _dt.datetime | None) -> str:
    return t.strftime("%Y-%m-%d %H:%M") if t is not None else ""


def _sidecar(d: str) -> dict:
    """The LIBRARY.json sidecar as a dict, {} when absent or unreadable (it is provenance only: the station never
    flies from it, and the integrity check below needs its two sha1s to be strings)."""
    try:
        with open(os.path.join(d, SIDECAR), "r", encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def _error_fit(why: str) -> dict:
    """An error entry fits NO slot (fail-closed: a picker that only reads ``fit`` still greys it out)."""
    return {low: f"unusable export — {why}" for low in RL_LOWS}


def _read_entry(d: str, where: str) -> Entry:
    name = export_name(d)
    base = {"name": name, "path": d, "rel": rel_to_repo(d), "where": where}
    side = _sidecar(d)
    source = str(side.get("source_dir") or "")

    def err(why: str, **kw) -> Entry:
        kw.setdefault("no_fly", no_fly_reason(name, kw.get("npz_sha1") or None))
        return Entry(**base, source=source, error=why, fit=_error_fit(why), **kw)

    try:
        with open(os.path.join(d, REQUIRED_FILES[0]), "rb") as f:
            raw = f.read()            # hashed AND parsed from the same bytes
    except OSError as e:
        return err(f"obs_spec.json unreadable ({e.strerror or e})")
    obs_sha1 = hashlib.sha1(raw).hexdigest()
    npz = os.path.join(d, REQUIRED_FILES[1])
    if not os.path.isfile(npz):
        return err("no policy_weights.npz", obs_sha1=obs_sha1)
    try:
        npz_sha1 = file_sha1(npz)
    except OSError as e:
        return err(f"policy_weights.npz unreadable ({e.strerror or e})", obs_sha1=obs_sha1)
    ids = {"obs_sha1": obs_sha1, "npz_sha1": npz_sha1, "id": export_id(obs_sha1, npz_sha1)}
    try:
        spec = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return err(f"obs_spec.json is not valid JSON ({e})", **ids)
    if not isinstance(spec, dict):
        return err("obs_spec.json is not a JSON object", **ids)
    # immutability (module docstring): a library entry whose files are not the bytes its sidecar recorded at sync
    # was modified in place — never offered (the same name could otherwise fly different weights than a past meta)
    for key, got, what in (("obs_sha1", obs_sha1, "obs_spec.json"), ("npz_sha1", npz_sha1, "policy_weights.npz")):
        want = side.get(key)
        if isinstance(want, str) and want and want.lower() != got:
            return err(f"{what} changed since the sync ({SIDECAR} {key} {want[:12]}, file {got[:12]}) — a library "
                       f"entry is never modified; sync the export again (it gets a new name)", **ids)
    mode = str(spec.get("action_mode", "axes"))
    _, out = _blk(spec, "output")
    out = out or {}
    pm = out.get("pwm_model")
    cap = None
    if mode == "pwm" and isinstance(pm, dict):
        try:                                      # rl_pwm.PwmModel: float(pwm_model.get("pwm_cap_trained", 1.0))
            cap = float(pm.get("pwm_cap_trained", 1.0))
            cap = cap if math.isfinite(cap) else None
        except (TypeError, ValueError):
            cap = None
    ck = spec.get("checkpoint")
    ck = ck if isinstance(ck, str) else ""
    it = spec.get("rsl_rl_iteration")
    if isinstance(it, bool) or not isinstance(it, int):
        it = side.get("rsl_rl_iteration")
        if isinstance(it, bool) or not isinstance(it, int):
            m = _MODEL_RE.search(ck)
            it = int(m.group(1)) if m else None
    trained = _fmt_stamp(checkpoint_run(ck)[1])
    if not trained and isinstance(side.get("run_stamp"), str):
        try:
            trained = _fmt_stamp(_dt.datetime.fromisoformat(side["run_stamp"]))
        except ValueError:
            trained = ""
    _, ar = _blk(spec, "attitude_reference")
    _, vs = _blk(spec, "velocity_source")
    _, oi = _blk(spec, "obs_integral")
    dr = spec.get("dr_preset")
    return Entry(**base, **ids, action_mode=mode, att_trained=bool(ar.get("trained")) if ar else False,
                 output_kind=str(out.get("kind") or ""), cap_trained=cap,
                 dr_preset=dr if isinstance(dr, str) else "", iteration=it, checkpoint=ck, trained=trained,
                 vel_trained=bool(vs) and vs.get("trained") is True, integral=bool(oi) and oi.get("enabled") is True,
                 fit=slot_fit(spec), no_fly=no_fly_reason(name, npz_sha1, ck), source=source, error="")


def _entry(d: str, where: str) -> Entry:
    key = (where, _stat_key(os.path.join(d, REQUIRED_FILES[0])), _stat_key(os.path.join(d, REQUIRED_FILES[1])),
           _stat_key(os.path.join(d, SIDECAR)))
    hit = _CACHE.get(d)
    if hit is not None and hit[0] == key:
        return hit[1]
    e = _read_entry(d, where)
    _CACHE[d] = (key, e)
    return e


def _roots(roots) -> list:
    """[(where, abs root)]: LIBRARY_ROOTS by default; a bare path is labelled by the LIBRARY_ROOTS root it IS, else
    "other"; a (where, path) pair keeps its label."""
    if roots is None:
        roots = LIBRARY_ROOTS
    elif isinstance(roots, (str, os.PathLike)):
        roots = [roots]
    known = {resolve_dir(p): w for w, p in LIBRARY_ROOTS}
    out = []
    for r in roots:
        if isinstance(r, (tuple, list)) and len(r) == 2:
            out.append((str(r[0]), resolve_dir(r[1])))
        else:
            a = resolve_dir(r)
            out.append((known.get(a, "other"), a))
    return out


def scan(roots=None) -> list[Entry]:
    """Every export under the roots (default LIBRARY_ROOTS, in priority order; each root's entries sorted by folder
    name): each IMMEDIATE subdir holding obs_spec.json. Hidden / temp dirs ('.'-prefixed — the sync tool's
    '.tmp-<name>-<pid>' while it copies) are skipped; dirs without obs_spec.json are not exports and are ignored; a
    missing root (data/rl_policies before the first sync) lists nothing. An export that cannot be read (bad JSON, no
    npz, files changed since the sync) is an Entry with ``error`` set — listed, never a candidate unless asked.
    Cached per path on (mtime_ns, size, inode) of the two files and the sidecar."""
    out = []
    for where, rdir in _roots(roots):
        try:
            with os.scandir(rdir) as it:
                names = sorted(e.name for e in it if not e.name.startswith(".") and e.is_dir())
        except OSError:
            continue
        for name in names:
            d = os.path.join(rdir, name)
            if os.path.isfile(os.path.join(d, REQUIRED_FILES[0])):
                out.append(_entry(d, where))
    return out


def dedupe(entries) -> list[Entry]:
    """The FIRST entry per id, in the given order — scan() lists the station root first, so an export present in both
    roots is listed as its station copy. Error entries are all kept (and never hide a readable one)."""
    seen, out = set(), []
    for e in entries:
        if e.error or not e.id:
            out.append(e)
            continue
        if e.id in seen:
            continue
        seen.add(e.id)
        out.append(e)
    return out


def refusal_reason(entry, low, cap_ctx=None) -> str | None:
    """Why ``entry`` must not be OFFERED for RL LOW ``low``, or None — the ONE rule candidates(), the picker dialog
    (rov_gui/widgets/rl_policy_dialog.py refusal()) and `./c3 rl-policies list` share (2026-11-02: before it, the
    three kept their own copies, and the iteration floor below would have had to be added to each). First hit wins:
      1. unreadable (entry.error)                — "unusable export — <error>" (= _error_fit's text)
      2. the flight ban (entry.no_fly)           — NO_FLY's text (+ how a copy was recognised)
      3. the iteration floor (MIN_ITERATION)     — a smoke export passes slot_fit AND the follower build (its obs_spec
                                                   is complete), so only this keeps a hand-copied one out of the picker;
                                                   an unknown iteration (None) is left to the build
      4. the slot pre-filter (entry.fit[low])    — a MISSING verdict is a refusal (fail-closed), as it always was
      5. the pwm transport cap (2026-11-02)      — ONLY with ``cap_ctx`` (MpcStatus.rl_pwm_cap: the worker's cap in
                                                   force, its sink ceiling, each pwm LOW's allow_caps_below_trained),
                                                   for the pwm LOWs and an entry whose cap_trained is known: _cap_refusal
    The text is what the picker shows; never "" for a refusal. A ``low`` outside RL_LOWS has no verdict, so it is
    refused (candidates() raises for it instead). Without ``cap_ctx`` (None / {}) — candidates(), the CLI — it is
    exactly the rule of before (1-4)."""
    if entry.error:
        return f"unusable export — {entry.error}"
    if entry.no_fly is not None:                # not just truthiness: only no_fly_reason's None means "not banned"
        return str(entry.no_fly) or "flight ban"
    if entry.iteration is not None and entry.iteration < MIN_ITERATION:
        return f"smoke-length export ({entry.iteration} iterations < {MIN_ITERATION})"
    why = (entry.fit or {}).get(low, "not evaluated for this LOW")
    if why is not None:
        return str(why) or "not evaluated for this LOW"
    return _cap_refusal(entry, low, cap_ctx)


#: the RL LOWs on the pwm transport — they SHARE its cap rl_pwm.pwm_cap (workers.MpcWorker._RL_PWM_SLOTS)
_PWM_LOWS = ("rl_pwm", "rl_pwm_7d")


def _finite(v) -> float | None:
    """float(v) for a finite real number (never a bool / str), else None."""
    if v is None or isinstance(v, (bool, str, bytes)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _cap_refusal(entry, low, cap_ctx) -> str | None:
    """refusal_reason's step 5 (2026-11-02, the LOW row's cap ▾): what the pwm LOW's BUILD would refuse at the
    transport cap in force, said before the pick — the build stays the judge (set_rl_policy re-runs it). Judged only
    for a pwm LOW, an entry with a known cap_trained and a ``cap_ctx`` carrying a finite "cap" (None otherwise):
      (a) cap_trained > cap and the LOW's allow_below is false — HwRlPwm / HwRlPwm7d refuse a cap below trained
          without the LOW's recorded opt-out;
      (b) the sink's ceiling known and min(cap, ceiling) != min(cap, cap_trained) — the thruster sink would enforce
          another window than the follower flies (workers._sink_gate_compare refuses that pick).
    The VERDICT is exactly (a) or (b). The remedy names the cap at which the export flies its trained cap — clipped to
    the sink's ceiling when that is lower (set_rl_pwm_cap refuses a cap above it). One case has no remedy on the LOW
    row: a LOW that refuses caps below trained, an export trained ABOVE the sink's ceiling — below its trained cap it
    is (a), at or above it the sink's window is lower (b) — and the text says so instead of naming a cap that the next
    look would refuse."""
    if not isinstance(cap_ctx, dict) or not cap_ctx or low not in _PWM_LOWS:
        return None
    t = _finite(getattr(entry, "cap_trained", None))
    cap = _finite(cap_ctx.get("cap"))
    if t is None or cap is None:
        return None
    allow = cap_ctx.get("allow_below")
    allow = bool(allow.get(low, False)) if isinstance(allow, dict) else False
    ceiling = _finite(cap_ctx.get("ceiling"))
    ceiling = ceiling if (ceiling is not None and ceiling > 0.0) else None
    label = LOW_LABELS.get(low, low)
    below = t > cap + 1e-9 and not allow                                                    # (a)
    window = ceiling is not None and abs(min(cap, ceiling) - min(cap, t)) > 1e-9            # (b)
    if not (below or window):
        return None
    if not allow and ceiling is not None and t > ceiling + 1e-9:
        return (f"trained at cap {t:g}, above the thruster sink's window {ceiling:g}, and {label} refuses caps below "
                f"trained — no LOW-row cap flies it here")
    if below:
        return (f"trained at cap {t:g} > the transport cap {cap:g} and {label} refuses caps below trained — set the "
                f"LOW row's cap to {t:g}")
    return (f"flies cap {min(cap, t):g} but the thruster sink's window is {min(cap, ceiling):g} — set the LOW row's "
            f"cap to {min(t, ceiling):g}")


def candidates(entries, low, include_refused=False) -> list[Entry]:
    """The entries LOW ``low`` can be offered: refusal_reason(e, low) None (readable, not banned, not smoke-length,
    fit[low] None) — or, with include_refused, every entry (the picker greys the rest out). Newest ``trained`` first,
    then by name; an unknown stamp ("") sorts last. ValueError for a low outside RL_LOWS."""
    if low not in RL_LOWS:
        raise ValueError(f"unknown RL LOW {low!r} (one of {list(RL_LOWS)})")
    out = [e for e in entries if include_refused or refusal_reason(e, low) is None]
    out.sort(key=lambda e: e.name)
    out.sort(key=lambda e: e.trained, reverse=True)
    return out
