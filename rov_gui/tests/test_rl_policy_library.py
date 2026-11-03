#!/usr/bin/env python3
"""test_rl_policy_library.py — `./c3 rl-policies` (rov_gui/tools/rl_policy_library.py, 2026-11-01): the sync that
copies RL_controller exports into the git-ignored library data/rl_policies/, and the list the LOW row's picker offers.

Every test builds a FAKE RL_controller tree and a fake library in tmp_path from a real station export's files
(rov_gui/control/rl_policies/pwm16N_s3, read only) — the real RL_controller and data/rl_policies are never touched.

What is pinned, in the order a failure would hurt:
  * an entry is NEVER overwritten, modified or deleted: a name held by other bytes (library, station folder, or an
    earlier source of the same sync) gets '<name>@<YYYYMMDD-HHMMSS>'; both held -> skipped; a second sync is all "up
    to date" and leaves every inode and mtime as it was — also after a Policies/ export's mtime (its '@stamp') moved,
    because "up to date" looks at the whole '<tag>@*' family; a failure mid-copy leaves no temp folder and no entry;
  * every skip rule, with its reason: '_' / '.' folders, no stamp, unnamed runs, smoke runs, no exported/, a missing
    required file, a file newer than --settle-s, a run export whose obs_spec.json is older than its npz (an interrupted
    re-export; Policies/ exempt), bad JSON, rsl_rl_iteration absent or < --min-iteration (default
    rl_library.MIN_ITERATION), an npz the station's own loader refuses, inconsistent dims, non-finite weights;
  * a dry run writes nothing anywhere; the RL_controller tree is never written; --dst is refused (inside the station
    folder / RL_controller / the repo outside data/) on the typed path AND through any symlink;
  * what an entry holds: obs_spec + npz (+ onnx / README*.md, never policy.pt), the LIBRARY.json sidecar (provenance +
    the two sha1s rl_library.scan verifies), every file 0444;
  * a flight-banned export is imported (DESIGN 2026-11-01 §5) but never offered (by name or by a renamed copy's npz);
  * `list` = rl_library.dedupe(scan()) / candidates(), the refused ones only with --all, each with why (the same
    rl_library.refusal_reason the picker uses — a smoke-length entry says so);
  * the script runs by PATH from any cwd without Qt, and `./c3 rl-policies` routes to it by path — never
    `-m rov_gui...`, which RL_controller's GPU guard would read as the station and SIGSTOP training for.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest -q -p no:cacheprovider \\
        rov_gui/tests/test_rl_policy_library.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import rl_library as L                      # noqa: E402
from rov_gui.tools import rl_policy_library as T                 # noqa: E402

TOOL = ROOT / "rov_gui" / "tools" / "rl_policy_library.py"
STATION = ROOT / "rov_gui" / "control" / "rl_policies"
#: pwm, attitude-trained, rsl_rl_iteration 2399, obs 136 = frame_dim 34 x history_len 4, 8 outputs, 3 layers
BASE = STATION / "pwm16N_s3"
#: every fixture file is this old (local time) — past any --settle-s; the '@stamp' of a Policies/ export
OLD = dt.datetime(2026, 10, 19, 7, 0, 0).timestamp()
RUNS = Path("logs") / "rsl_rl" / "rov_posehold"


# ----------------------------------------------------------------------------- fixtures
def _spec(folder: str | None = None, **over) -> dict:
    """pwm16N_s3's obs_spec; with a run folder, its checkpoint points into that run (as export_policy.py writes it)."""
    s = json.loads((BASE / "obs_spec.json").read_text(encoding="utf-8"))
    if folder is not None:
        s["checkpoint"] = f"logs/rsl_rl/rov_posehold/{folder}/model_{s['rsl_rl_iteration']}.pt"
    s.update(over)
    return s


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _base_arrays() -> dict:
    with np.load(BASE / "policy_weights.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _age(p: Path, t: float = OLD) -> None:
    for q in [p, *p.rglob("*")]:
        os.utime(q, (t, t), follow_symlinks=False)


def _export(d: Path, spec: dict | None = None, npz: bytes | None = None, extra: dict | None = None,
            spec_text: str | None = None) -> Path:
    """In export_policy.py's write order — npz, the rest, obs_spec.json LAST — so an un-aged one never looks like an
    interrupted re-export (obs_spec.json older than the npz: T.SKIP_INTERRUPTED)."""
    d.mkdir(parents=True, exist_ok=True)
    text = spec_text if spec_text is not None else json.dumps(spec if spec is not None else _spec(), indent=2)
    (d / "policy_weights.npz").write_bytes(npz if npz is not None else (BASE / "policy_weights.npz").read_bytes())
    for k, v in (extra or {}).items():
        (d / k).write_bytes(v)
    (d / "obs_spec.json").write_text(text, encoding="utf-8")
    return d


def _run(rl: Path, folder: str, spec: dict | None = None, env_yaml: str | None = None, age: bool = True,
         **kw) -> Path:
    """<rl>/logs/rsl_rl/rov_posehold/<folder>/exported/ (+ params/env.yaml), aged past --settle-s."""
    run = rl / RUNS / folder
    ex = _export(run / "exported", spec if spec is not None else _spec(folder), **kw)
    if env_yaml is not None:
        (run / "params").mkdir(parents=True, exist_ok=True)
        (run / "params" / "env.yaml").write_text(env_yaml, encoding="utf-8")
    if age:
        _age(run)
    return ex


def _policy(rl: Path, name: str, spec: dict | None = None, **kw) -> Path:
    d = _export(rl / "Policies" / name, spec if spec is not None else _spec(
        checkpoint="logs/rsl_rl/rov_posehold/2026-10-09_12-49-17_abl_base_pwm_s0/model_1199.pt"), **kw)
    _age(d)
    return d


def _tree(p: Path) -> dict:
    """{relative path: (is_dir, size, mtime_ns, mode, inode)} of everything under p — a write anywhere shows."""
    out = {}
    for q in [p, *sorted(p.rglob("*"))]:
        st = q.lstat()
        out[str(q.relative_to(p))] = (q.is_dir(), st.st_size if not q.is_dir() else 0, st.st_mtime_ns,
                                      stat.S_IMODE(st.st_mode), st.st_ino)
    return out


def _by_tag(items) -> dict:
    out = {}
    for i in items:
        out.setdefault(i.tag, []).append(i)
    return out


def _cli(capsys, *argv) -> tuple[int, str]:
    rc = T.main([str(a) for a in argv])
    return rc, capsys.readouterr().out


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- skip rules
def test_every_skip_rule_has_its_reason(tmp_path):
    rl = tmp_path / "RL_controller"
    runs = rl / RUNS
    (runs / "_aborted_20261026_reboot" / "2026-10-26_17-28-10_pwm17F_s2").mkdir(parents=True)
    _export(runs / ".hidden_run" / "exported")
    _export(runs / "notes" / "exported")                                   # no run stamp
    _run(rl, "2026-09-21_13-31-02")                                        # rsl_rl's unnamed smoke folder
    _run(rl, "2026-10-03_12-07-37_smoke_att7_axes6", spec=_spec(rsl_rl_iteration=1))   # complete-looking export
    (runs / "2026-10-15_20-22-58_pwm11att12C_s0" / "params").mkdir(parents=True)        # aborted: no exported/
    ex = _run(rl, "2026-10-16_01-00-00_nonpz_s0")
    (ex / "policy_weights.npz").unlink()
    ex = _run(rl, "2026-10-16_01-00-01_nospec_s0")
    (ex / "obs_spec.json").unlink()
    _run(rl, "2026-10-16_01-00-02_fresh_s0", age=False)                     # mtime now: still being written
    _run(rl, "2026-10-16_01-00-03_badjson_s0", spec_text="{")
    _run(rl, "2026-10-16_01-00-04_lowit_s0", spec=_spec(rsl_rl_iteration=49))
    s = _spec("2026-10-16_01-00-05_noit_s0")
    del s["rsl_rl_iteration"]
    _run(rl, "2026-10-16_01-00-05_noit_s0", spec=s)
    _run(rl, "2026-10-16_01-00-06_badnpz_s0", npz=b"PK not a zip at all")
    _run(rl, "2026-10-16_01-00-07_dimsobs_s0", spec=_spec(obs_dim=999))
    _run(rl, "2026-10-16_01-00-08_dimsw0_s0", spec=_spec(frame_dim=33, obs_dim=132))   # W0 takes 136
    _run(rl, "2026-10-16_01-00-09_dimsact_s0", spec=_spec(num_actions=6, act_dim=6))   # last layer gives 8
    nan = _base_arrays()
    nan["W1"] = nan["W1"].copy()
    nan["W1"][3, 5] = np.nan
    _run(rl, "2026-10-16_01-00-10_nan_s0", npz=_npz_bytes(**nan))
    _run(rl, "2026-10-16_01-00-11_good_s0")
    ex = _run(rl, "2026-10-16_01-00-12_interrupted_s0")
    os.utime(ex / "policy_weights.npz", (OLD, OLD + 60))                    # new npz, then the re-export died
    (runs / "README.txt").write_text("a loose file is not a run")
    _policy(rl, "pwm_k0_s0")
    (rl / "Policies" / "hw_mpc_josh_k0.yaml").write_text("rl_pwm: {}\n")    # not an export: not mentioned
    (rl / "Policies" / "empty").mkdir()                                     # no obs_spec.json: not an export

    items = T.plan(str(rl), str(tmp_path / "lib"), station=None)
    got = {i.tag: (i.action, i.reason) for i in items}
    assert got == {
        "_aborted_20261026_reboot": (T.SKIP, T.SKIP_NOT_RUN), ".hidden_run": (T.SKIP, T.SKIP_NOT_RUN),
        "notes": (T.SKIP, T.SKIP_NO_STAMP), "2026-09-21_13-31-02": (T.SKIP, T.SKIP_NO_TAG),
        "smoke_att7_axes6": (T.SKIP, T.SKIP_SMOKE), "pwm11att12C_s0": (T.SKIP, T.SKIP_NO_EXPORT),
        "nonpz_s0": (T.SKIP, T.SKIP_MISSING), "nospec_s0": (T.SKIP, T.SKIP_MISSING),
        "fresh_s0": (T.SKIP, T.SKIP_SETTLING), "interrupted_s0": (T.SKIP, T.SKIP_INTERRUPTED),
        "badjson_s0": (T.SKIP, T.SKIP_JSON),
        "lowit_s0": (T.SKIP, T.SKIP_ITERATION), "noit_s0": (T.SKIP, T.SKIP_ITERATION),
        "badnpz_s0": (T.SKIP, T.SKIP_NPZ), "dimsobs_s0": (T.SKIP, T.SKIP_DIMS), "dimsw0_s0": (T.SKIP, T.SKIP_DIMS),
        "dimsact_s0": (T.SKIP, T.SKIP_DIMS), "nan_s0": (T.SKIP, T.SKIP_NONFINITE),
        "good_s0": (T.IMPORT, ""), "pwm_k0_s0": (T.IMPORT, ""),
    }
    by = {i.tag: i for i in items}
    assert by["nonpz_s0"].detail == "no policy_weights.npz" and by["nospec_s0"].detail == "no obs_spec.json"
    assert "rsl_rl_iteration 49 < 50" in by["lowit_s0"].detail
    assert "obs_dim 999" in by["dimsobs_s0"].detail
    assert "first layer takes 136 inputs" in by["dimsw0_s0"].detail
    assert "last layer gives 8 outputs != num_actions 6" in by["dimsact_s0"].detail
    assert "--settle-s 120" in by["fresh_s0"].detail
    assert "BEFORE policy_weights.npz" in by["interrupted_s0"].detail
    # the summary counts every reason (and the CLI prints each one)
    n, per = T.counts(items)
    assert n == {T.IMPORT: 2, T.UP_TO_DATE: 0, T.SKIP: 18, T.FAILED: 0}
    assert per[T.SKIP_NOT_RUN] == 2 and per[T.SKIP_MISSING] == 2 and per[T.SKIP_DIMS] == 3
    assert per[T.SKIP_ITERATION] == 2 and list(per) == [r for r in T.SKIP_ORDER if r in per]


def test_the_cli_summary_counts_each_reason(tmp_path, capsys):
    rl = tmp_path / "rl"
    _run(rl, "2026-10-03_12-07-37_smoke_pwm11")
    _run(rl, "2026-09-21_13-31-02")
    _run(rl, "2026-10-16_01-00-04_lowit_s0", spec=_spec(rsl_rl_iteration=2))
    _run(rl, "2026-10-16_01-00-11_good_s0")
    rc, out = _cli(capsys, "sync", "--dry-run", "--rl-root", rl, "--dst", tmp_path / "lib",
                   "--station", tmp_path / "st")
    assert rc == 0
    assert "DRY RUN: nothing is written" in out
    assert re.search(r"^would import \(1\):", out, re.M) and re.search(r"^  good_s0 +RL_PWM,RL_PWM_7D +0\.5 +2399 ",
                                                                       out, re.M)
    assert (f"summary: would import 1 · up to date 0 · skipped 3 ({T.SKIP_NO_TAG} 1 · {T.SKIP_SMOKE} 1 · "
            f"{T.SKIP_ITERATION} 1)") in out
    assert "lowit_s0 (rsl_rl_iteration 2 < 50)" in out
    assert "of those, offered to RL 0 · RL_PWM 1 · RL_7D 0 · RL_PWM_7D 1" in out


def test_settle_window_and_min_iteration_are_knobs(tmp_path):
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    _run(rl, "2026-10-16_01-00-02_fresh_s0", age=False)
    _run(rl, "2026-10-16_01-00-03_young_s0", spec=_spec(rsl_rl_iteration=10))
    (i1, i2) = T.plan(str(rl), str(lib))
    assert (i1.reason, i2.reason) == (T.SKIP_SETTLING, T.SKIP_ITERATION)
    (i1, i2) = T.plan(str(rl), str(lib), settle_s=0.0, min_iteration=10)
    assert (i1.action, i2.action) == (T.IMPORT, T.IMPORT)
    # a file stamped in the FUTURE (clock skew) counts as being written — fail-closed
    fut = rl / RUNS / "2026-10-16_01-00-03_young_s0" / "exported" / "policy.onnx"
    fut.write_bytes(b"x")
    os.utime(fut, (OLD, T.time.time() + 3600))
    (_, i2) = T.plan(str(rl), str(lib), settle_s=0.0, min_iteration=10)
    assert i2.reason == T.SKIP_SETTLING and "policy.onnx" in i2.detail


def test_an_interrupted_re_export_is_skipped(tmp_path):
    """2026-11-02 (review): export_policy.py writes the npz first and obs_spec.json last, so a run export whose obs_spec
    is OLDER than its npz is a re-export that died in between (new weights, old checkpoint / iteration labels) — never
    imported, however old. Policies/ hand copies are exempt (the real two: obs_spec 42-67 ms older than the npz)."""
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    ex = _run(rl, "2026-10-16_01-00-12_reexp_s0")
    os.utime(ex / "policy_weights.npz", (OLD, OLD + 60))
    pol = _policy(rl, "pwm_k0_s0")
    os.utime(pol / "policy_weights.npz", (OLD, OLD + 0.067))
    (it, ip) = T.plan(str(rl), str(lib))
    assert (it.action, it.reason, ip.action) == (T.SKIP, T.SKIP_INTERRUPTED, T.IMPORT)
    assert it.detail.startswith("obs_spec.json modified 2026-10-19 07:00:00.000, BEFORE policy_weights.npz "
                                "2026-10-19 07:01:00.000 — "), it.detail
    # while it is still being written, the settle window speaks first
    os.utime(ex / "policy_weights.npz", None)
    assert T.plan(str(rl), str(lib))[0].reason == T.SKIP_SETTLING
    # the export finished (obs_spec.json written last): it comes in
    os.utime(ex / "policy_weights.npz", (OLD, OLD + 60))
    os.utime(ex / "obs_spec.json", (OLD, OLD + 61))
    assert T.plan(str(rl), str(lib))[0].action == T.IMPORT


def test_the_iteration_floor_is_rl_library_min_iteration(tmp_path, capsys, monkeypatch):
    """--min-iteration's default IS rl_library.MIN_ITERATION — the picker's own floor (rl_library.refusal_reason) —
    read when the CLI runs, never a second literal; a lower --min-iteration says out loud that what comes in between is
    never offered."""
    import inspect
    assert inspect.signature(T.plan).parameters["min_iteration"].default == L.MIN_ITERATION
    assert inspect.signature(T.sync).parameters["min_iteration"].default == L.MIN_ITERATION
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    _run(rl, "2026-10-16_01-00-04_lowit_s0", spec=_spec(rsl_rl_iteration=L.MIN_ITERATION - 1))
    _run(rl, "2026-10-16_01-00-05_atfloor_s0", spec=_spec(rsl_rl_iteration=L.MIN_ITERATION))
    args = ("sync", "--dry-run", "--rl-root", rl, "--dst", lib, "--station", tmp_path / "st")
    rc, out = _cli(capsys, *args)
    assert rc == 0 and f"rsl_rl_iteration >= {L.MIN_ITERATION};" in out and "NOTE" not in out
    assert f"lowit_s0 (rsl_rl_iteration {L.MIN_ITERATION - 1} < {L.MIN_ITERATION})" in out
    assert re.search(r"^  atfloor_s0 +RL_PWM,RL_PWM_7D ", out, re.M)
    monkeypatch.setattr(L, "MIN_ITERATION", 7)
    rc, out = _cli(capsys, *args)
    assert rc == 0 and "rsl_rl_iteration >= 7;" in out and "would import (2):" in out
    rc, out = _cli(capsys, *args, "--min-iteration", 3)
    assert "NOTE     --min-iteration 3 is below rl_library.MIN_ITERATION 7" in out


@pytest.mark.parametrize("arrays, reason, needle", [
    (lambda a: {**a, "activation": np.array("swish")}, T.SKIP_NPZ, "KeyError"),
    (lambda a: {k: v for k, v in a.items() if k != "W2"}, T.SKIP_NPZ, "KeyError"),
    (lambda a: {**a, "W1": np.array([object()], dtype=object)}, T.SKIP_NPZ, "allow_pickle"),
    (lambda a: {**a, "W1": a["W1"][:, :60]}, T.SKIP_DIMS, "does not chain"),              # 64 -> 60: no chain
    (lambda a: {**a, "b0": a["b0"][:10]}, T.SKIP_DIMS, "does not chain"),
    (lambda a: {**a, "obs_mean": np.zeros(135), "obs_var": np.ones(135)}, T.SKIP_DIMS, "obs_mean shape"),
    (lambda a: {**a, "obs_mean": np.zeros(136), "obs_var": np.r_[np.ones(135), 0.0]}, T.SKIP_NONFINITE, "obs_var"),
    (lambda a: {**a, "obs_mean": np.zeros(136), "obs_var": -np.ones(136)}, T.SKIP_NONFINITE, ""),
    (lambda a: {**a, "b2": np.full(8, np.inf)}, T.SKIP_NONFINITE, "NaN / inf"),
    # every weight finite, but the forward pass overflows (act() would clip the inf to a silent full command)
    (lambda a: {**a, "b0": np.full(64, 1e308), "W1": np.full((64, 64), 10.0)}, T.SKIP_NONFINITE, "forward pass"),
    (lambda a: {**a, "obs_mean": np.zeros(136), "obs_var": np.ones(136)}, "", ""),         # a normalizer is fine
])
def test_check_weights_uses_the_station_loader(arrays, reason, needle):
    got = T.check_weights(_spec(), _npz_bytes(**arrays(_base_arrays())))
    assert got[0] == reason and needle in got[1], got
    assert T.check_weights(_spec(), (BASE / "policy_weights.npz").read_bytes()) == ("", "")
    assert T.check_weights({"frame_dim": 34}, (BASE / "policy_weights.npz").read_bytes())[0] == T.SKIP_DIMS


# ----------------------------------------------------------------------------- names, never overwriting
def test_names_stamps_collisions_and_idempotence(tmp_path, capsys):
    rl, lib, st = tmp_path / "rl", tmp_path / "lib", tmp_path / "station"
    other = _spec(dr_preset="other bytes")                                 # same weights, other obs_spec = other id
    # a re-used tag (the imminent cfd v2 rerun of pwm11att14Rcfd_s1): the OLDER run keeps the plain name
    _run(rl, "2026-10-22_11-30-15_alpha_s0", spec=_spec("2026-10-22_11-30-15_alpha_s0", dr_preset="v2"))
    _run(rl, "2026-10-20_10-00-00_alpha_s0")
    # the library already holds beta_s0 with other bytes; gamma_s0 AND gamma_s0@<its stamp> too
    _export(lib / "beta_s0", other)
    _run(rl, "2026-10-21_09-00-00_beta_s0")
    _export(lib / "gamma_s0", other)
    _export(lib / "gamma_s0@20261021-100000", _spec(dr_preset="yet other"))
    _run(rl, "2026-10-21_10-00-00_gamma_s0")
    # the station folder (read only) holds delta_s0 = the same bytes, eps_s0 = other bytes
    _run(rl, "2026-10-21_11-00-00_delta_s0")
    shutil.copytree(rl / RUNS / "2026-10-21_11-00-00_delta_s0" / "exported", st / "delta_s0")
    _run(rl, "2026-10-21_12-00-00_eps_s0")
    _export(st / "eps_s0", other)
    # a Policies/ export named like a run tag: its '@stamp' is its obs_spec.json mtime
    _policy(rl, "alpha_s0", spec=_spec(dr_preset="josh"))
    _age(lib)
    _age(st)
    before_lib, before_st = _tree(lib), _tree(st)

    rc, out = _cli(capsys, "sync", "--rl-root", rl, "--dst", lib, "--station", st)
    assert rc == 0, out
    items = _by_tag(T.plan(str(rl), str(lib), str(st)))                    # what a SECOND sync would do
    want_names = {"alpha_s0", "alpha_s0@20261022-113015", "beta_s0@20261021-090000", "eps_s0@20261021-120000",
                  "alpha_s0@20261019-070000"}
    entries = {p.name for p in lib.iterdir()}
    assert entries == want_names | {"beta_s0", "gamma_s0", "gamma_s0@20261021-100000"}, out
    assert not (lib / "delta_s0").exists()                                 # the station has it: not copied again
    assert "gamma_s0" in out and T.SKIP_NAME in out
    # nothing that existed was touched: same inode, bytes, mtime, mode — in the library (its own folder '.' gains
    # entries, so only what was IN it counts) and in the station folder
    after_lib = _tree(lib)
    assert {k: v for k, v in after_lib.items() if k in before_lib and k != "."} == \
        {k: v for k, v in before_lib.items() if k != "."}
    assert _tree(st) == before_st
    # the second sync: everything up to date, nothing written
    assert {(t, i.action) for t, its in items.items() for i in its} == {
        ("alpha_s0", T.UP_TO_DATE), ("beta_s0", T.UP_TO_DATE), ("gamma_s0", T.SKIP), ("delta_s0", T.UP_TO_DATE),
        ("eps_s0", T.UP_TO_DATE)}
    assert sorted(i.name for i in items["alpha_s0"]) == ["alpha_s0", "alpha_s0@20261019-070000",
                                                         "alpha_s0@20261022-113015"]
    assert items["delta_s0"][0].held_by == L.rel_to_repo(st / "delta_s0")
    rc, out2 = _cli(capsys, "sync", "--rl-root", rl, "--dst", lib, "--station", st)
    assert rc == 0 and "summary: imported 0 · up to date 6 · skipped 1" in out2
    assert _tree(lib) == after_lib
    # 2026-11-02 (review): a Policies/ '@stamp' is obs_spec.json's mtime — a touch (a git checkout round trip, a cp
    # without -p) moves it, yet the SAME bytes are already in as alpha_s0@20261019-070000: up to date through the tag
    # family, never imported again as alpha_s0@20261019-080000
    os.utime(rl / "Policies" / "alpha_s0" / "obs_spec.json", (OLD, OLD + 3600))
    (pol,) = [i for i in T.plan(str(rl), str(lib), str(st)) if i.src is not None and i.src.run is None]
    assert (pol.action, pol.name, pol.held_by) == (T.UP_TO_DATE, "alpha_s0@20261019-070000",
                                                   L.rel_to_repo(lib / "alpha_s0@20261019-070000"))
    rc, out3 = _cli(capsys, "sync", "--rl-root", rl, "--dst", lib, "--station", st)
    assert rc == 0 and "summary: imported 0 · up to date 6 · skipped 1" in out3
    assert _tree(lib) == after_lib


def test_a_dry_run_names_like_the_real_one_and_writes_nothing(tmp_path, capsys):
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    _run(rl, "2026-10-20_10-00-00_alpha_s0")
    _run(rl, "2026-10-22_11-30-15_alpha_s0", spec=_spec("2026-10-22_11-30-15_alpha_s0", dr_preset="v2"))
    _policy(rl, "pwm_k0_s0")
    before = _tree(tmp_path)
    rc, out = _cli(capsys, "sync", "--dry-run", "--rl-root", rl, "--dst", lib, "--station", tmp_path / "st")
    assert rc == 0 and _tree(tmp_path) == before and not lib.exists()
    dry = [i.name for i in T.plan(str(rl), str(lib)) if i.action == T.IMPORT]
    assert dry == ["alpha_s0", "alpha_s0@20261022-113015", "pwm_k0_s0"]   # the dry run remembers its own plan
    assert "would import (3):" in out and "nothing written" in out
    rc, out = _cli(capsys, "sync", "--rl-root", rl, "--dst", lib, "--station", tmp_path / "st")
    assert rc == 0 and sorted(p.name for p in lib.iterdir()) == sorted(dry)


def test_the_source_tree_is_never_written(tmp_path, capsys):
    rl = tmp_path / "rl"
    _run(rl, "2026-10-20_10-00-00_alpha_s0", env_yaml="hydro_model: cfd\n")
    _run(rl, "2026-10-20_10-00-01_lowit_s0", spec=_spec(rsl_rl_iteration=3))
    _policy(rl, "pwm_k0_s0")
    before = _tree(rl)
    rc, _ = _cli(capsys, "sync", "--rl-root", rl, "--dst", tmp_path / "lib", "--station", tmp_path / "st")
    assert rc == 0 and _tree(rl) == before


# ----------------------------------------------------------------------------- what an entry holds
def test_sidecar_files_and_read_only_entry(tmp_path):
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    folder = "2026-10-30_19-41-13_pwm11att14Rcfd_s1"
    env = ("viewer:\n  eye: !!python/tuple\n  - 7.5\n  hydro_model: nested-not-top-level\n"
           "hydro_model: cfd\ncfd_drag_path: '/home/x/RL_controller/configs/cfd_drag.yaml'\n"
           "fossen_added_mass: !!python/tuple\n- 5.5\n")
    ex = _run(rl, folder, env_yaml=env, extra={"policy.onnx": b"onnx", "policy_wrench.onnx": b"wonnx",
                                                "README.md": b"# notes", "README.en.md": b"# en",
                                                "policy.pt": b"torchscript", "notes.txt": b"x"})
    _policy(rl, "pwm_k0_s0")
    t0 = dt.datetime.now().astimezone()
    items = T.sync(str(rl), str(lib))
    assert [i.action for i in items] == [T.IMPORT, T.IMPORT]
    e = lib / "pwm11att14Rcfd_s1"
    assert sorted(p.name for p in e.iterdir()) == sorted(
        ["obs_spec.json", "policy_weights.npz", "policy.onnx", "policy_wrench.onnx", "README.en.md", "README.md",
         L.SIDECAR])                                                       # never policy.pt / other files
    for p in e.iterdir():
        assert stat.S_IMODE(p.stat().st_mode) == 0o444, p.name
        if p.name != L.SIDECAR:
            assert p.read_bytes() == (ex / p.name).read_bytes()
    side = json.loads((e / L.SIDECAR).read_text(encoding="utf-8"))
    obs, npz = _sha1(ex / "obs_spec.json"), _sha1(ex / "policy_weights.npz")
    assert side == {
        "schema": 1, "name": "pwm11att14Rcfd_s1", "tag": "pwm11att14Rcfd_s1", "source_dir": str(ex),
        "source_run": folder, "run_stamp": "2026-10-30T19:41:13", "stamp_from": "run folder",
        "synced_at": side["synced_at"],
        "files": {p.name: _sha1(p) for p in e.iterdir() if p.name != L.SIDECAR},
        "obs_sha1": obs, "npz_sha1": npz, "id": L.export_id(obs, npz), "rsl_rl_iteration": 2399,
        "checkpoint": f"logs/rsl_rl/rov_posehold/{folder}/model_2399.pt", "dr_preset": "pwm15", "action_mode": "pwm",
        "attitude_trained": True, "cap_trained": 0.5,
        "hydro": {"hydro_model": "cfd", "cfd_drag_path": "/home/x/RL_controller/configs/cfd_drag.yaml"}}
    synced = dt.datetime.fromisoformat(side["synced_at"])
    assert synced.tzinfo is not None and abs((synced - t0).total_seconds()) < 60
    pol = json.loads((lib / "pwm_k0_s0" / L.SIDECAR).read_text(encoding="utf-8"))
    assert (pol["source_run"], pol["stamp_from"], pol["run_stamp"], pol["hydro"]) == \
        (None, "obs_spec mtime", "2026-10-19T07:00:00", None)
    assert pol["source_dir"] == str(rl / "Policies" / "pwm_k0_s0")
    # rl_library reads the entry as the picker will: provenance, no error, the same id
    ents = {x.name: x for x in L.scan([("library", lib)])}
    x = ents["pwm11att14Rcfd_s1"]
    assert (x.error, x.source, x.id, x.where) == ("", str(ex), L.export_id(obs, npz), "library")
    assert x.trained == "2026-10-30 19:41" and x.fit["rl_pwm_7d"] is None
    assert not list(lib.glob(".tmp-*"))


def test_read_hydro(tmp_path):
    run = tmp_path / "run"
    assert T.read_hydro(None) is None and T.read_hydro(str(run)) is None
    (run / "params").mkdir(parents=True)
    (run / "params" / "env.yaml").write_text("sim:\n  hydro_model: nested\nfoo: !!python/tuple\n- 1\n")
    assert T.read_hydro(str(run)) is None
    (run / "params" / "env.yaml").write_text('hydro_model: "inertia_box"\ncfd_drag_path: null\n')
    assert T.read_hydro(str(run)) == {"hydro_model": "inertia_box", "cfd_drag_path": None}


# ----------------------------------------------------------------------------- failures leave nothing behind
def test_a_failed_copy_leaves_no_temp_and_no_entry(tmp_path, monkeypatch, capsys):
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    _run(rl, "2026-10-20_10-00-00_alpha_s0", extra={"policy.onnx": b"onnx"})
    _run(rl, "2026-10-20_10-00-01_beta_s0")
    real = shutil.copyfile

    def flaky(src, dst, *a, **kw):
        if "alpha_s0" in str(src) and str(src).endswith("policy.onnx"):
            raise OSError(28, "No space left on device")
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(T.shutil, "copyfile", flaky)
    rc, out = _cli(capsys, "sync", "--rl-root", rl, "--dst", lib, "--station", tmp_path / "st")
    assert rc == 1 and "FAILED (1)" in out and "No space left on device" in out
    assert sorted(p.name for p in lib.iterdir()) == ["beta_s0"]           # the other export still came in
    assert not list(lib.glob(".tmp-*"))


def test_an_export_that_changes_while_copied_is_not_imported(tmp_path, monkeypatch):
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    _run(rl, "2026-10-20_10-00-00_alpha_s0")
    real = shutil.copyfile

    def swap(src, dst, *a, **kw):
        r = real(src, dst, *a, **kw)
        if str(dst).endswith("obs_spec.json"):
            Path(dst).write_text(json.dumps(_spec(dr_preset="rewritten mid-copy")), encoding="utf-8")
        return r

    monkeypatch.setattr(T.shutil, "copyfile", swap)
    (it,) = T.sync(str(rl), str(lib))
    assert it.action == T.FAILED and "changed while it was copied" in it.detail
    assert list(lib.iterdir()) == []


def test_a_name_that_appears_meanwhile_is_never_overwritten(tmp_path, monkeypatch):
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    _run(rl, "2026-10-20_10-00-00_alpha_s0")
    (it,) = T.plan(str(rl), str(lib))
    # (a) it exists by the time install() starts: refused before any temp folder
    (lib / "alpha_s0").mkdir(parents=True)
    (lib / "alpha_s0" / "intruder").write_text("mine")
    with pytest.raises(FileExistsError):
        T.install(it, str(lib))
    assert sorted(p.name for p in lib.iterdir()) == ["alpha_s0"]
    shutil.rmtree(lib / "alpha_s0")
    # (b) it appears between the last check and the rename (another sync): rename() fails, the temp goes
    real = os.rename

    def racing(src, dst, *a, **kw):
        if str(dst).endswith("alpha_s0"):
            os.mkdir(dst)
            Path(dst, "intruder").write_text("mine")
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(T.os, "rename", racing)
    with pytest.raises(OSError):
        T.install(it, str(lib))
    monkeypatch.undo()
    assert sorted(p.name for p in lib.iterdir()) == ["alpha_s0"]
    assert [p.name for p in (lib / "alpha_s0").iterdir()] == ["intruder"]


# ----------------------------------------------------------------------------- the station folder is a namespace
def test_the_real_station_folder_names_are_never_reused(tmp_path):
    """Read-only on the real station folder: a byte-identical export is 'up to date' there (not copied again); a
    same-name export with other bytes — pwm12PI_s0's RL_controller obs_spec differs from the station copy
    (KNOWN_ISSUES.md:811-813) — comes in under its '@stamp'."""
    rl, lib = tmp_path / "rl", tmp_path / "lib"
    folder = "2026-10-19_03-25-36_pwm16N_s3"
    _run(rl, folder, spec=None, spec_text=(BASE / "obs_spec.json").read_text(encoding="utf-8"))
    pi = STATION / "pwm12PI_s0"
    s = json.loads((pi / "obs_spec.json").read_text(encoding="utf-8"))
    s["obs_integral"] = {k: v for k, v in s["obs_integral"].items() if k != "cond_att_rad"}
    _run(rl, "2026-10-18_05-01-44_pwm12PI_s0", spec=s, npz=(pi / "policy_weights.npz").read_bytes())
    before = _tree(STATION)
    items = {i.tag: i for i in T.plan(str(rl), str(lib), station=str(STATION))}
    assert (items["pwm16N_s3"].action, items["pwm16N_s3"].held_by) == \
        (T.UP_TO_DATE, "rov_gui/control/rl_policies/pwm16N_s3")
    assert (items["pwm12PI_s0"].action, items["pwm12PI_s0"].name) == (T.IMPORT, "pwm12PI_s0@20261018-050144")
    assert _tree(STATION) == before


# ----------------------------------------------------------------------------- the flight ban
def test_banned_exports_are_imported_but_never_offered(tmp_path, capsys):
    rl, lib, st = tmp_path / "rl", tmp_path / "lib", tmp_path / "st"
    p10 = STATION / "pwm10_s1"
    spec10 = json.loads((p10 / "obs_spec.json").read_text(encoding="utf-8"))
    # by name (pwm10_s2 has no station copy) and as a RENAMED copy of pwm10_s1's weights under an innocent tag
    _run(rl, "2026-09-27_11-06-05_pwm10_s2", spec={**spec10, "checkpoint": "x/model_1199.pt"},
         npz=(p10 / "policy_weights.npz").read_bytes())
    _run(rl, "2026-10-20_10-00-00_fresh_s0", spec={**spec10, "checkpoint": "y/model_1199.pt"},
         npz=(p10 / "policy_weights.npz").read_bytes())
    rc, out = _cli(capsys, "sync", "--rl-root", rl, "--dst", lib, "--station", st)
    assert rc == 0 and re.search(r"^  pwm10_s2 +NO-FLY ", out, re.M) and re.search(r"^  fresh_s0 +NO-FLY ", out, re.M)
    assert "no-fly 2 (listed greyed, refused at the pick)" in out
    ents = L.dedupe(L.scan([("library", lib)]))
    assert {e.name for e in ents} == {"pwm10_s2", "fresh_s0"} and all(e.no_fly for e in ents)
    for low in L.RL_LOWS:
        assert L.candidates(ents, low) == []
    rc, out = _cli(capsys, "list", "--station", st, "--dst", lib)
    assert "pwm10_s2" not in out and "(2 more not listed" in out
    rc, out = _cli(capsys, "list", "--station", st, "--dst", lib, "--all")
    assert re.search(r"^  fresh_s0 .*NO-FLY: flight ban — .*same weights as pwm10_s1", out, re.M)


# ----------------------------------------------------------------------------- list
def _rows(out: str) -> list[str]:
    """The NAME column of the printed table (rows indented two spaces, after the header row)."""
    lines = out.splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("  NAME "))
    names = []
    for ln in lines[hdr + 1:]:
        if not ln.startswith("  ") or ln.startswith("   "):
            break
        names.append(ln.split()[0])
    return names


@pytest.fixture
def two_roots(tmp_path):
    st, lib, rl = tmp_path / "station", tmp_path / "lib", tmp_path / "rl"
    for n in ("pwm16N_s3", "traj12_s2", "pwm11k0.2_s0", "pwm10_s1"):
        shutil.copytree(STATION / n, st / n)
    _run(rl, "2026-10-26_18-19-59_zeta_s0", spec=_spec("2026-10-26_18-19-59_zeta_s0"))
    _run(rl, "2026-10-18_05-02-04_wrench_s0", spec=_spec("2026-10-18_05-02-04_wrench_s0", action_mode="wrench6"))
    _run(rl, "2026-10-19_03-25-36_pwm16N_s3", spec_text=(BASE / "obs_spec.json").read_text(encoding="utf-8"))
    assert all(i.action in (T.IMPORT, T.UP_TO_DATE) for i in T.sync(str(rl), str(lib), str(st)))
    _export(lib / "broken", spec_text="{")                                 # an unreadable entry
    s = json.loads((STATION / "traj12_s2" / "obs_spec.json").read_text(encoding="utf-8"))
    s["dr_preset"] = "hand-edited"
    _export(lib / "traj12_s2", s, npz=(STATION / "traj12_s2" / "policy_weights.npz").read_bytes())
    return st, lib


def test_list_offers_what_some_low_flies(two_roots, capsys):
    st, lib = two_roots
    rc, out = _cli(capsys, "list", "--station", st, "--dst", lib)
    assert rc == 0
    assert out.startswith(f"RL policy library — station {st} (4) · library {lib} (4) · 8 after dedupe")
    # newest trained first; no-fly, no-LOW and unreadable ones only with --all
    assert _rows(out) == ["zeta_s0", "pwm16N_s3", "pwm11k0.2_s0", "traj12_s2", "traj12_s2"]
    assert "(3 more not listed — no-fly, no fitting LOW, smoke-length or unreadable; --all lists them with why)" in out
    row = next(ln for ln in out.splitlines() if ln.startswith("  zeta_s0 "))
    assert row.split()[1:4] == ["library", "2026-10-26", "18:19"] and "RL_PWM,RL_PWM_7D" in row
    assert " 0.5 " in row and " pwm15 " in row and " 2399 " in row and "DVL blend" in row
    # two different exports under one name: legal (a hand copy), but said out loud
    assert re.search(r"^NOTE: 'traj12_s2' names 2 different exports \(library id [0-9a-f]{12}, station id "
                     r"[0-9a-f]{12}\) — tell them apart by id$", out, re.M)
    rc, out = _cli(capsys, "list", "--station", st, "--dst", lib, "--all")
    assert sorted(_rows(out)) == sorted(["zeta_s0", "pwm16N_s3", "pwm11k0.2_s0", "traj12_s2", "traj12_s2",
                                         "pwm10_s1", "wrench_s0", "broken"])
    assert re.search(r"^  wrench_s0 .*no LOW: action_mode wrench6 is not flown by the station$", out, re.M)
    assert re.search(r"^  broken .*ERROR: obs_spec.json is not valid JSON", out, re.M)
    assert re.search(r"^  pwm10_s1 .*NO-FLY: flight ban", out, re.M)


@pytest.mark.parametrize("low", L.RL_LOWS)
def test_list_low_is_exactly_rl_library_candidates(two_roots, capsys, low):
    st, lib = two_roots
    ents = L.dedupe(L.scan([("station", st), ("library", lib)]))
    for arg in (low, L.LOW_LABELS[low]):                                   # the key or the panel label
        rc, out = _cli(capsys, "list", "--station", st, "--dst", lib, "--low", arg)
        want = [e.name for e in L.candidates(ents, low)]
        assert rc == 0
        if want:
            assert _rows(out) == want
        else:                                                              # rl_7d: no export flies it (no table)
            assert "  NAME " not in out
        assert f"{L.LOW_LABELS[low]} ({low}): {len(want)} offered by the LOW row's picker" in out
        assert f"({len(ents) - len(want)} more not listed — no-fly, not for {L.LOW_LABELS[low]}, smoke-length or " \
               f"unreadable" in out
    rc, out = _cli(capsys, "list", "--station", st, "--dst", lib, "--low", low, "--all")
    assert _rows(out) == [e.name for e in L.candidates(ents, low, include_refused=True)]
    if low == "rl_pwm_7d":
        assert re.search(r"^  traj12_s2 .*not for RL_PWM_7D: axes export — RL only$", out, re.M)
        assert re.search(r"^  pwm11k0.2_s0 .*not for RL_PWM_7D: level-trained — no roll/pitch reference$", out, re.M)


def test_list_refuses_a_smoke_length_entry_like_the_picker(two_roots, capsys):
    """A smoke export hand-copied into the library (complete obs_spec, rsl_rl_iteration 2) fits the pwm slots and would
    build — rl_library.refusal_reason's iteration floor keeps it out of the picker, and `list` says exactly that."""
    st, lib = two_roots
    _export(lib / "smoky_s0", _spec(rsl_rl_iteration=2))
    floor = f"smoke-length export (2 iterations < {L.MIN_ITERATION})"
    ents = L.dedupe(L.scan([("station", st), ("library", lib)]))
    smoky = next(e for e in ents if e.name == "smoky_s0")
    assert smoky.fit["rl_pwm_7d"] is None and T.offered_to(smoky) == []
    rc, out = _cli(capsys, "list", "--station", st, "--dst", lib)
    assert rc == 0 and "smoky_s0" not in out
    rc, out = _cli(capsys, "list", "--station", st, "--dst", lib, "--all")
    assert re.search(rf"^  smoky_s0 .*no LOW: {re.escape(floor)}$", out, re.M), out
    for low in L.RL_LOWS:
        rc, out = _cli(capsys, "list", "--station", st, "--dst", lib, "--low", low, "--all")
        assert _rows(out) == [e.name for e in L.candidates(ents, low, include_refused=True)]
        assert re.search(rf"^  smoky_s0 .*not for {L.LOW_LABELS[low]}: {re.escape(floor)}$", out, re.M), out
        rc, out = _cli(capsys, "list", "--station", st, "--dst", lib, "--low", low)
        assert "smoky_s0" not in out and (_rows(out) if "  NAME " in out else []) == \
            [e.name for e in L.candidates(ents, low)]


def test_list_rejects_an_unknown_low(capsys):
    with pytest.raises(SystemExit) as e:
        T.main(["list", "--low", "mpc"])
    assert e.value.code == 2 and "is not an RL LOW" in capsys.readouterr().err


# ----------------------------------------------------------------------------- guards
def test_sync_refuses_a_destination_it_must_not_write(tmp_path, capsys):
    rl = tmp_path / "rl"
    _run(rl, "2026-10-20_10-00-00_alpha_s0")
    before = _tree(tmp_path)
    for dst, station, needle in ((tmp_path / "st", tmp_path / "st", "station folder"),
                                 (tmp_path / "st" / "sub", tmp_path / "st", "station folder"),
                                 (rl / "lib", tmp_path / "st", "read only"),
                                 ("rov_gui/control/lib", tmp_path / "st", "git-ignored data/")):
        with pytest.raises(SystemExit) as e:
            T.main(["sync", "--rl-root", str(rl), "--dst", str(dst), "--station", str(station)])
        assert needle in str(e.value.code), (dst, e.value.code)
        assert "through a symlink" not in str(e.value.code)                # a text hit keeps its 2026-11-01 wording
    with pytest.raises(SystemExit, match="neither logs/rsl_rl/rov_posehold/ nor Policies/"):
        T.main(["sync", "--rl-root", str(tmp_path / "nowhere"), "--dst", str(tmp_path / "lib")])
    with pytest.raises(SystemExit, match=">= 0"):
        T.main(["sync", "--rl-root", str(rl), "--dst", str(tmp_path / "lib"), "--settle-s", "-1"])
    assert _tree(tmp_path) == before
    assert not (ROOT / "rov_gui" / "control" / "lib").exists()


def test_sync_refuses_a_destination_reached_through_a_symlink(tmp_path, capsys, monkeypatch):
    """2026-11-02 (review): the three refusals hold on PHYSICAL paths too — a --dst that is (or sits under) a symlink
    into the station folder or the RL_controller tree, an --rl-root given through an alias with --dst inside its real
    path, and a repo whose data/ is a symlink into RL_controller all passed the text-only checks, and install() would
    have written through the link. A data/ symlinked to another disk is still a valid library home."""
    rl, st = tmp_path / "rl", tmp_path / "st"
    _run(rl, "2026-10-20_10-00-00_alpha_s0")
    (rl / "Policies").mkdir()
    st.mkdir()
    (tmp_path / "to_station").symlink_to(st, target_is_directory=True)
    (tmp_path / "to_logs").symlink_to(rl / "logs", target_is_directory=True)
    (tmp_path / "rl_alias").symlink_to(rl, target_is_directory=True)
    repo = tmp_path / "repo"                                               # a stand-in repo (REPO_ROOT) ...
    (repo / "rov_gui" / "control").mkdir(parents=True)
    (repo / "data").symlink_to(rl / "Policies", target_is_directory=True)  # ... whose data/ "shares" Policies/
    (tmp_path / "to_src").symlink_to(repo / "rov_gui", target_is_directory=True)
    monkeypatch.setattr(L, "REPO_ROOT", repo)
    monkeypatch.setattr(L, "_banned_npz_cache", L._banned_npz_cache)       # restored whatever the stand-in caches
    before = _tree(tmp_path)
    for dst, rl_root, needle in ((tmp_path / "to_station", rl, "station folder"),
                                 (tmp_path / "to_station" / "sub", rl, "station folder"),
                                 (tmp_path / "to_logs" / "lib", rl, "read only"),
                                 (rl / "lib", tmp_path / "rl_alias", "read only"),
                                 (repo / "data" / "rl_policies", rl, "read only"),
                                 (tmp_path / "to_src" / "control", rl, "git-ignored data/")):
        with pytest.raises(SystemExit) as e:
            T.main(["sync", "--rl-root", str(rl_root), "--dst", str(dst), "--station", str(st)])
        msg = str(e.value.code)
        assert needle in msg and " through a symlink (" in msg, (dst, msg)
        assert msg.startswith(f"rl-policies sync: --dst {dst} "), msg              # the path as typed comes first
    assert _tree(tmp_path) == before
    # a data/ symlinked to another disk is a valid library home (its realpath is the allowed subtree)
    (repo / "data").unlink()
    (tmp_path / "otherdisk" / "data").mkdir(parents=True)
    (repo / "data").symlink_to(tmp_path / "otherdisk" / "data", target_is_directory=True)
    rc, out = _cli(capsys, "sync", "--dry-run", "--rl-root", rl, "--dst", repo / "data" / "rl_policies",
                   "--station", st)
    assert rc == 0 and "would import (1):" in out
    assert not (tmp_path / "otherdisk" / "data" / "rl_policies").exists()


def test_default_rl_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RL_CONTROLLER_ROOT", str(tmp_path / "rlc"))
    assert T.default_rl_root() == str(tmp_path / "rlc")
    monkeypatch.delenv("RL_CONTROLLER_ROOT")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert T.default_rl_root() == str(tmp_path / "Desktop" / "RL_controller")


# ----------------------------------------------------------------------------- how it is launched
_GPU_GUARD = re.compile(r"^[^ ]*python[0-9.]* (.* )?-m rov_gui")   # RL_controller scripts/_gpu_guard.sh:13 (pgrep -f)


def test_the_script_runs_by_path_from_any_cwd_without_qt(tmp_path):
    rl, lib, st = tmp_path / "rl", tmp_path / "lib", tmp_path / "st"
    _run(rl, "2026-10-20_10-00-00_alpha_s0")
    shutil.copytree(STATION / "traj12_s2", st / "traj12_s2")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [sys.executable, str(TOOL), "sync", "--rl-root", str(rl), "--dst", str(lib), "--station", str(st)]
    assert _GPU_GUARD.search(" ".join(argv)) is None
    r = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "summary: imported 1" in r.stdout and (lib / "alpha_s0" / L.SIDECAR).is_file()
    r = subprocess.run([sys.executable, str(TOOL), "list", "--station", str(st), "--dst", str(lib)], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and _rows(r.stdout) == ["alpha_s0", "traj12_s2"], r.stdout + r.stderr
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); from rov_gui.tools import rl_policy_library as T; "
             "rc = T.main(['sync', '--dry-run', '--rl-root', sys.argv[2], '--dst', sys.argv[3], '--station', "
             "sys.argv[4]]) or T.main(['list', '--station', sys.argv[4], '--dst', sys.argv[3]]); "
             "print('QT', sorted(m for m in sys.modules if m.split('.')[0] in "
             "('PyQt5', 'PyQt6', 'PySide2', 'PySide6'))); sys.exit(rc)")
    r = subprocess.run([sys.executable, "-c", probe, str(ROOT), str(rl), str(lib), str(st)], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "QT []" in r.stdout, r.stdout + r.stderr


def test_c3_routes_rl_policies_to_the_script_by_path(tmp_path):
    text = (ROOT / "c3").read_text(encoding="utf-8")
    arm = re.search(r"\n  rl-policies\)\n(.*?)\n    ;;\n", text, re.S)
    assert arm, "./c3 has no rl-policies) arm"
    body = arm.group(1)
    assert 'cd "$REPO"' in body and 'exec "$PY" rov_gui/tools/rl_policy_library.py "$@"' in body
    assert text.index("\n  rl-policies)\n") < text.index("\n  *)              mod=\"$tool\" ;;")   # before the fallback
    assert not re.search(r"-m\s+rov_gui\.tools\.rl_policy_library", text)
    assert re.search(r"^  rl-policies +\S", text, re.M), "./c3 --help does not list rl-policies"
    assert re.search(r"^#\s+\./c3 rl-policies ", text, re.M), "the ./c3 header does not show rl-policies"
    # and it runs: same interpreter as this test (C3_PY), from a foreign cwd
    st, lib = tmp_path / "st", tmp_path / "lib"
    shutil.copytree(STATION / "pwm16N_s3", st / "pwm16N_s3")
    env = {**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "C3_PY": sys.executable}
    r = subprocess.run(["bash", str(ROOT / "c3"), "rl-policies", "list", "--low", "rl_pwm_7d", "--station", str(st),
                        "--dst", str(lib)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert _rows(r.stdout) == ["pwm16N_s3"] and "RL_PWM_7D (rl_pwm_7d): 1 offered" in r.stdout
    r = subprocess.run(["bash", str(ROOT / "c3"), "--help"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "rl-policies" in r.stdout
