#!/usr/bin/env python3
"""test_rl_library.py — the RL policy library (control/rl_library.py, 2026-11-01): what the LOW row's picker lists,
how an export is identified, and which RL LOW it is offered to.

What is pinned, in the order a failure would hurt:
  * the flight ban holds by name, by '<name>@<stamp>', by a RENAMED copy of a banned station export (weights sha1) and
    by the training run the obs_spec checkpoint names; a banned export is never a candidate;
  * refusal_reason — the one "may it be offered" rule of candidates(), the picker dialog and `./c3 rl-policies list`:
    unreadable -> flight ban -> fewer than MIN_ITERATION training iterations (a smoke export) -> slot_fit; the floor
    changes nothing for any station export;
  * slot_fit never offers an export to a LOW whose follower refuses it for an export-intrinsic reason: an explicit
    table for EVERY station export (every export in the folder needs a row; a row whose folder is not on this machine
    — an export the operator has not committed, on a clean checkout — is SKIPPED with that message, never failed),
    dry builds of the real followers (HwRl / HwRlPwm / HwRl7d / HwRlPwm7d, opt-outs forced like the 2026-11-01
    agreement check) on the station exports and on synthetic defects of them;
  * identity is content: sha1 of obs_spec.json / policy_weights.npz, id = sha1("<obs>:<npz>")[:12]; a library entry
    whose files no longer match its sidecar is an error entry;
  * paths: a relative policy_dir is REPO_ROOT-relative (never the cwd), '~' is expanded, nothing is realpath'd;
    rel_to_repo never climbs '../..'; '<stamp>_<tag>/exported' is named by its tag;
  * scan / dedupe / candidates: error entries for unreadable exports, '.'-prefixed dirs skipped, dirs without
    obs_spec.json ignored, the station copy kept on a duplicate, newest first, include_refused, the per-path cache
    re-reads a changed file;
  * LOW_LABELS = the panel's MODE_LABELS for the four RL keys.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest -q -p no:cacheprovider \\
        rov_gui/tests/test_rl_library.py
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import rl_library as L                      # noqa: E402

STATION = ROOT / "rov_gui" / "control" / "rl_policies"
#: what the module computes paths against: REPO_ROOT is abspath-based (no realpath), ROOT above is resolve()d — the
#: same folder on this machine (test_repo_root_is_rl_policy_repo_root), but path EQUALITY is checked against this one
REPO = L.REPO_ROOT
STATION_ABS = REPO / "rov_gui" / "control" / "rl_policies"


def _bump_mtime(p: Path) -> None:
    """A rewrite inside one tick of the filesystem's coarse clock can keep (mtime_ns, size): move the mtime on
    explicitly, as a later write would (the cache key is taken from the files, never from the test's intent)."""
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

_PWM_ONLY = "pwm export — RL_PWM / RL_PWM_7D only"
_AXES_ONLY = "axes export — RL only"
_AXES6_ONLY = "axes6 export — RL_7D only"
_LEVEL = "level-trained — no roll/pitch reference"
_PWM_ATT = {"rl": _PWM_ONLY, "rl_pwm": None, "rl_7d": _PWM_ONLY, "rl_pwm_7d": None}
_PWM_LEVEL = {"rl": _PWM_ONLY, "rl_pwm": None, "rl_7d": _PWM_ONLY, "rl_pwm_7d": _LEVEL}
_AXES = {"rl": None, "rl_pwm": _AXES_ONLY, "rl_7d": _AXES_ONLY, "rl_pwm_7d": _AXES_ONLY}
#: slot_fit of EVERY station export, pinned 2026-11-01 from their obs_spec.json and the dry-build agreement check
#: (DESIGN §1a: all 15 build exactly where this says None, with rl_pwm / rl_pwm_7d opt-outs forced). rl_pwm flies the
#: attitude-trained pwm exports too — LEVEL (HwRl._q_des); only rl_pwm_7d tracks their roll/pitch.
EXPECTED_FIT = {
    "att7pwm_s0": _PWM_ATT, "pwm10_s1": _PWM_LEVEL,
    "pwm11att14R_s1": _PWM_ATT, "pwm11att14Rcfd_s1": _PWM_ATT, "pwm11att14Rw_s1": _PWM_ATT,
    "pwm11att7_s0": _PWM_ATT, "pwm11att9X_s0": _PWM_ATT, "pwm11k0.2_s0": _PWM_LEVEL,
    "pwm12PI_s0": _PWM_ATT, "pwm14B_s0": _PWM_ATT, "pwm15N_s0": _PWM_ATT, "pwm16N_s3": _PWM_ATT,
    "traj10_s1": _AXES, "traj12_s2": _AXES, "traj5_s2": _AXES,
}
#: the station exports that are banned from flight (RL_controller README §13.8) — kept on disk for the Lua install
#: tool and the tests
EXPECTED_NO_FLY = {"att7pwm_s0", "pwm10_s1"}
# 2026-11-02 (review): pwm11att14Rw_s1 / pwm11att14Rcfd_s1 were NOT committed when these rows were written (`git
# status`: '?? rov_gui/control/rl_policies/pwm11att14Rw_s1/' and '...Rcfd_s1/', 2026-11-01) — a checkout without them
# must not fail here. So THOSE TWO rows (_UNCOMMITTED) are checked only where their folder is on this machine
# (_present / _skip_if_absent: the repo's "not on this machine" skip, as test_rl_integral_obs.py:1278 does for an
# export outside git), and every fixture that copies a station export uses a git-tracked one. A missing TRACKED export
# still fails (it was deleted — restore it or drop its row), and every export that IS in the station folder needs its
# row (test_every_station_export_has_a_row).
_UNCOMMITTED = frozenset({"pwm11att14Rw_s1", "pwm11att14Rcfd_s1"})


# ----------------------------------------------------------------------------- helpers
def _spec(name: str) -> dict:
    return json.loads((STATION / name / "obs_spec.json").read_text(encoding="utf-8"))


def _on_disk(name: str) -> bool:
    return (STATION / name / "obs_spec.json").is_file()


def _present() -> list[str]:
    """The EXPECTED_FIT rows to check, sorted: every tracked row (a deleted one then fails where it is read) and the
    _UNCOMMITTED rows whose station export is on this machine."""
    return sorted(n for n in EXPECTED_FIT if n not in _UNCOMMITTED or _on_disk(n))


def _skip_if_absent(name: str) -> None:
    if _on_disk(name):
        return
    assert name in _UNCOMMITTED, (f"station export {name} is git-tracked but missing ({STATION / name}/obs_spec.json "
                                  f"absent) — restore it, or drop its EXPECTED_FIT row if it was removed on purpose")
    pytest.skip(f"station export {name} is not on this machine ({STATION / name}/obs_spec.json absent — an export "
                f"the operator has not committed, on a clean checkout?): its EXPECTED_FIT row is not checked")


def _write_export(d: Path, spec: dict, npz_from: Path | None = None, npz_arrays: dict | None = None,
                  sidecar: dict | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "obs_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    if npz_from is not None:
        shutil.copyfile(npz_from, d / "policy_weights.npz")
    elif npz_arrays is not None:
        with open(d / "policy_weights.npz", "wb") as f:
            np.savez(f, **npz_arrays)
    if sidecar is not None:
        (d / L.SIDECAR).write_text(json.dumps(sidecar), encoding="utf-8")
    return d


def _copy_station(name: str, d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for f in L.REQUIRED_FILES:
        shutil.copyfile(STATION / name / f, d / f)
    return d


def _mlp_npz(obs_dim: int, act_dim: int) -> dict:
    """A policy_weights.npz in RL_controller's export layout (NumpyPolicy: n_layers, W<i>/b<i>, activation)."""
    rng = np.random.default_rng(7)
    return {"W0": rng.normal(0.0, 0.1, (64, obs_dim)).astype(np.float32), "b0": np.zeros(64, np.float32),
            "W1": rng.normal(0.0, 0.1, (64, 64)).astype(np.float32), "b1": np.zeros(64, np.float32),
            "W2": rng.normal(0.0, 0.1, (act_dim, 64)).astype(np.float32), "b2": np.zeros(act_dim, np.float32),
            "n_layers": np.array(3), "activation": np.array("elu")}


def _axes6_spec(gv: bool = True) -> dict:
    """An attitude-trained 6-axis (rl_7d) export spec, derived from traj12_s2's (none ships in the station folder;
    RL_controller has only the smoke_att7_axes6 run, 2026-10-03)."""
    from rov_gui.control import rl_policy as RP
    s = _spec("traj12_s2")
    width = 17 + (4 if gv else 0) + 6
    s.update({"action_mode": "axes6", "num_actions": 6, "act_dim": 6, "obs_goal_velocity": gv,
              "frame_layout": RP.frame_layout(gv, 6), "frame_dim": width, "obs_dim": width * int(s["history_len"]),
              "attitude_reference": {"trained": True, "roll_max_rad": 0.17, "pitch_max_rad": 0.35,
                                     "rate_max_rad_s": 0.35},
              "attitude_cap_trained": [0.5, 0.75],
              "checkpoint": "logs/rsl_rl/rov_posehold/2026-10-03_12-07-37_synthetic_axes6/model_1199.pt"})
    return s


@functools.lru_cache(maxsize=1)
def _cfg0():
    from rov_gui.control.geometry import MpcConfig
    return MpcConfig.load(str(ROOT / "config" / "hw_mpc.yaml"))


#: Build refusals that depend on the CONFIG, not the export (DESIGN §1a: classified apart — the predicates cannot see
#: the config): the DVL blend vs velocity_source, rl_pwm.obs_guard's constants, ctrl_hz, the cap, the sign probe.
_CONFIG_MARKERS = ("was trained on the DVL-primary", "was trained with velocity_source.", "estimator.vel_lp_alpha",
                   "!= the station's rl_pwm.obs_guard", "rl_pwm.obs_guard.min_tags is", "but ctrl_hz is",
                   "below the policy's trained cap", "% of thruster", "thruster probe")


def _dry_build(low: str, export_dir) -> str | None:
    """The LOW's own follower built on a deepcopy of hw_mpc.yaml with only its policy_dir replaced, the cap opt-outs
    forced true and the sign probe off (exactly the agreement check's config) -> None (built) or the refusal text."""
    from rov_gui.control import rl_policy as RP
    cls = {"rl": RP.HwRl, "rl_pwm": RP.HwRlPwm, "rl_7d": RP.HwRl7d, "rl_pwm_7d": RP.HwRlPwm7d}[low]
    cfg = copy.deepcopy(_cfg0())
    blk = dict(getattr(cfg, low) or {})
    blk["policy_dir"] = str(export_dir)
    if low in ("rl_pwm", "rl_7d", "rl_pwm_7d"):
        blk["allow_caps_below_trained"] = True
    setattr(cfg, low, blk)
    tr = dict(cfg.rl_pwm or {})
    tr.update(allow_caps_below_trained=True, require_thruster_probe=False)
    cfg.rl_pwm = tr
    try:
        cls(cfg, log=lambda m: None)
    except Exception as e:                                      # noqa: BLE001 — any refusal is a refusal
        return f"{type(e).__name__}: {e}"
    return None


def _assert_agrees(export_dir, fit: dict, who: str) -> set:
    """A predicate ACCEPT builds (or is refused only for a config reason); a predicate REFUSE is refused by the build.
    -> the LOWs whose follower actually BUILT (so a caller can tell a real pass from a config-refused one)."""
    built = set()
    for low in L.RL_LOWS:
        why = _dry_build(low, export_dir)
        if why is None:
            built.add(low)
        if fit[low] is None:
            assert why is None or any(m in why for m in _CONFIG_MARKERS), \
                f"{who}: slot_fit offers it to {low} but the build refuses it for an export reason: {why}"
        else:
            assert why is not None, f"{who}: slot_fit hides it from {low} ({fit[low]}) but the build accepts it"
    return built


# ----------------------------------------------------------------------------- shared names
def test_shared_names():
    assert L.RL_LOWS == ("rl", "rl_pwm", "rl_7d", "rl_pwm_7d")
    assert L.LOW_LABELS == {"rl": "RL", "rl_pwm": "RL_PWM", "rl_7d": "RL_7D", "rl_pwm_7d": "RL_PWM_7D"}
    assert L.LIBRARY_ROOTS == (("station", "rov_gui/control/rl_policies"), ("library", "data/rl_policies"))
    assert L.REQUIRED_FILES == ("obs_spec.json", "policy_weights.npz")
    assert L.SIDECAR == "LIBRARY.json"
    assert set(L.NO_FLY) == {"pwm10_s1", "pwm10_s2", "att7pwm_s0", "att7pwm_s1", "att7pwm_s2"}
    assert all(r.startswith("flight ban — ") and "§13.8" in r for r in L.NO_FLY.values())


def test_repo_root_is_rl_policy_repo_root():
    from rov_gui.control import rl_policy as RP
    assert isinstance(L.REPO_ROOT, Path)
    assert str(L.REPO_ROOT) == RP.REPO_ROOT
    assert (L.REPO_ROOT / "rov_gui" / "control" / "rl_library.py").is_file()
    assert L.REPO_ROOT.resolve() == ROOT


def test_low_labels_are_the_panel_mode_labels():
    from rov_gui.widgets.trajectory import MODE_LABELS
    assert {k: MODE_LABELS[k] for k in L.RL_LOWS} == L.LOW_LABELS


# ----------------------------------------------------------------------------- hashes and ids
def test_hashes_and_id(tmp_path):
    d = _write_export(tmp_path / "x", {"a": 1}, npz_arrays={"z": np.arange(3)})
    obs = hashlib.sha1((d / "obs_spec.json").read_bytes()).hexdigest()
    npz = hashlib.sha1((d / "policy_weights.npz").read_bytes()).hexdigest()
    assert L.file_sha1(d / "obs_spec.json") == obs
    assert L.export_hashes(d) == (obs, npz)
    assert L.export_hashes(str(d)) == (obs, npz)
    eid = L.export_id(obs, npz)
    assert eid == hashlib.sha1(f"{obs}:{npz}".encode()).hexdigest()[:12]
    assert len(eid) == 12 and all(c in "0123456789abcdef" for c in eid)
    # both halves count: same npz, another obs_spec = another export
    assert L.export_id(L.file_sha1(STATION / "traj12_s2" / "obs_spec.json"), npz) != eid
    # a relative export dir is the REPO_ROOT's, like policy_dir
    assert L.export_hashes("rov_gui/control/rl_policies/pwm16N_s3") == L.export_hashes(STATION / "pwm16N_s3")
    (d / "policy_weights.npz").unlink()
    with pytest.raises(FileNotFoundError):
        L.export_hashes(d)


def test_export_name():
    assert L.export_name("/x/y/pwm16N_s3") == "pwm16N_s3"
    assert L.export_name("/x/y/pwm16N_s3/") == "pwm16N_s3"
    assert L.export_name("/r/logs/rsl_rl/rov_posehold/2026-10-19_03-25-36_pwm16N_s3/exported") == "pwm16N_s3"
    assert L.export_name("/r/logs/rsl_rl/rov_posehold/2026-10-19_03-25-36_pwm16N_s3/exported/") == "pwm16N_s3"
    assert L.export_name("/r/2026-10-10_05-17-55_pwm11k0.2_s0/exported") == "pwm11k0.2_s0"
    assert L.export_name("/r/2026-10-30_19-41-13_pwm11att14Rcfd_s1/exported") == "pwm11att14Rcfd_s1"
    # an unnamed (smoke) run keeps its whole folder name; a non-stamp parent too
    assert L.export_name("/r/2026-09-21_13-31-02/exported") == "2026-09-21_13-31-02"
    assert L.export_name("/r/my_run/exported") == "my_run"
    assert L.export_name("/lib/pwm11att14Rcfd_s1@20261101-221500") == "pwm11att14Rcfd_s1@20261101-221500"
    assert L.export_name("rov_gui/control/rl_policies/traj12_s2") == "traj12_s2"
    assert L.export_name(STATION / "pwm11k0.2_s0") == "pwm11k0.2_s0"


def test_checkpoint_run():
    tag, st = L.checkpoint_run("logs/rsl_rl/rov_posehold/2026-10-19_03-25-36_pwm16N_s3/model_2399.pt")
    assert tag == "pwm16N_s3" and st.isoformat() == "2026-10-19T03:25:36"
    assert L.checkpoint_run("2026-10-09_12-49-17_abl_base_pwm_s0/model_1199")[0] == "abl_base_pwm_s0"
    assert L.checkpoint_run("model_1199.pt") == (None, None)
    assert L.checkpoint_run(None) == (None, None)
    assert L.checkpoint_run("logs/2026-13-45_99-99-99_bad/model_1.pt") == (None, None)


# ----------------------------------------------------------------------------- paths
def test_resolve_dir_is_repo_relative_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert L.resolve_dir("data/rl_policies/x") == str(REPO / "data" / "rl_policies" / "x")
    assert L.resolve_dir("rov_gui/control/rl_policies/pwm16N_s3/") == str(STATION_ABS / "pwm16N_s3")
    assert L.resolve_dir(Path("data") / "rl_policies") == str(REPO / "data" / "rl_policies")
    assert L.resolve_dir("/a/b/../c/") == "/a/c"
    assert not L.resolve_dir("x").startswith(str(tmp_path))


def test_resolve_dir_expands_home_and_keeps_symlinks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert L.resolve_dir("~/exp/pwm16N_s3") == str(tmp_path / "exp" / "pwm16N_s3")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert L.resolve_dir(link) == str(link)                    # lexical — the path the follower's meta records


def test_rel_to_repo(tmp_path):
    assert L.rel_to_repo(STATION_ABS / "pwm16N_s3") == "rov_gui/control/rl_policies/pwm16N_s3"
    assert L.rel_to_repo("data/rl_policies/pwm16N_s3") == "data/rl_policies/pwm16N_s3"
    assert L.rel_to_repo(REPO) == "."
    out = "/home/bdml/Desktop/RL_controller/logs/rsl_rl/rov_posehold/2026-10-19_03-25-36_pwm16N_s3/exported"
    assert L.rel_to_repo(out) == out
    assert L.rel_to_repo(tmp_path / "x") == str(tmp_path / "x")
    # a sibling folder sharing the repo's name as a PREFIX is outside (no string-prefix trap)
    sib = str(REPO.parent / (REPO.name + "_backup") / "x")
    assert L.rel_to_repo(sib) == sib
    assert ".." not in L.rel_to_repo(REPO.parent / "elsewhere")
    assert ".." not in L.rel_to_repo("../RL_controller/Policies/pwm_k0_s0")   # a '../' policy_dir is resolved first


def test_is_export_dir(tmp_path):
    d = _copy_station("pwm16N_s3", tmp_path / "e")
    assert L.is_export_dir(d) and L.is_export_dir(str(d))
    assert L.is_export_dir("rov_gui/control/rl_policies/pwm16N_s3")
    (d / "policy_weights.npz").unlink()
    assert not L.is_export_dir(d)
    assert not L.is_export_dir(d / "obs_spec.json")            # a file, not a directory
    assert not L.is_export_dir(tmp_path / "missing")
    assert not L.is_export_dir("")                              # = REPO_ROOT, which is no export


# ----------------------------------------------------------------------------- flight ban
def test_no_fly_by_name():
    for n, why in L.NO_FLY.items():
        assert L.no_fly_reason(n) == why
        assert L.no_fly_reason(f"{n}@20260927-110605") == why
    assert L.no_fly_reason(str(STATION / "pwm10_s1")) == L.NO_FLY["pwm10_s1"]
    assert L.no_fly_reason("/r/2026-09-27_11-06-05_pwm10_s2/exported") == L.NO_FLY["pwm10_s2"]
    for n in ("pwm16N_s3", "pwm10_s0", "pwm10_s10", "att7pwm", "pwm11att7_s0", "", None):
        assert L.no_fly_reason(n) is None, n
    # a non-banned name with a non-banned npz
    assert L.no_fly_reason("pwm16N_s3", L.export_hashes(STATION / "pwm16N_s3")[1]) is None


def test_no_fly_catches_a_renamed_copy(tmp_path):
    lib = tmp_path / "lib"
    _copy_station("pwm10_s1", lib / "fresh_pwm_policy")
    _copy_station("att7pwm_s0", lib / "good_one")
    npz = L.export_hashes(lib / "fresh_pwm_policy")[1]
    why = L.no_fly_reason("fresh_pwm_policy", npz)
    assert why is not None and why.startswith(L.NO_FLY["pwm10_s1"]) and "same weights as pwm10_s1" in why
    ents = {e.name: e for e in L.scan([("library", lib)])}
    assert set(ents) == {"fresh_pwm_policy", "good_one"}
    assert "pwm10_s1" in ents["fresh_pwm_policy"].no_fly and "att7pwm_s0" in ents["good_one"].no_fly
    for low in L.RL_LOWS:
        assert L.candidates(list(ents.values()), low) == []
    # the weights alone are enough: a renamed copy with an edited obs_spec (another checkpoint) is still caught
    s = _spec("pwm10_s1")
    s["checkpoint"] = "logs/rsl_rl/rov_posehold/2026-10-19_03-25-36_pwm16N_s3/model_2399.pt"
    d2 = _write_export(tmp_path / "lib2" / "renamed", s, npz_from=STATION / "pwm10_s1" / "policy_weights.npz")
    (e,) = L.scan([("library", d2.parent)])
    assert e.no_fly is not None and "same weights as pwm10_s1" in e.no_fly


def test_no_fly_by_the_checkpoint_run(tmp_path):
    """Other weights, but the obs_spec says they are the banned run's (another checkpoint / an edited re-export)."""
    s = _spec("pwm16N_s3")
    s["checkpoint"] = "logs/rsl_rl/rov_posehold/2026-09-27_11-06-05_pwm10_s2/model_900.pt"
    root = tmp_path / "lib"
    _write_export(root / "innocent", s, npz_from=STATION / "pwm16N_s3" / "policy_weights.npz")
    (e,) = L.scan([("library", root)])
    assert e.no_fly is not None and "trained as pwm10_s2" in e.no_fly and e.no_fly.startswith(L.NO_FLY["pwm10_s2"])
    assert L.no_fly_reason("innocent", e.npz_sha1, e.checkpoint) == e.no_fly
    assert L.candidates([e], "rl_pwm_7d") == [] and L.candidates([e], "rl_pwm_7d", include_refused=True) == [e]


# ----------------------------------------------------------------------------- slot_fit
def test_every_station_export_has_a_row():
    names = sorted(p.name for p in STATION.iterdir() if (p / "obs_spec.json").is_file())
    new = sorted(set(names) - set(EXPECTED_FIT))
    assert not new, (f"new station export(s) {new}: add a row to EXPECTED_FIT — derive it with rl_library.slot_fit "
                     f"and check it against a dry build of each RL LOW (test_slot_fit_agrees_with_the_build)")
    assert _present() == names                                  # (rows whose folder is absent: skipped below)


@pytest.mark.parametrize("name", sorted(EXPECTED_FIT))
def test_slot_fit_every_station_export(name):
    _skip_if_absent(name)
    assert L.slot_fit(_spec(name)) == EXPECTED_FIT[name], name


def test_slot_fit_agrees_with_the_build_on_the_station_exports():
    """DESIGN §1a on the station folder: a predicate ACCEPT is never an export-intrinsic build refusal, and nothing
    the build accepts is hidden (the 2026-11-01 sweep of every RL_controller export found the same: 0 / 0). The rows
    whose export is on this machine (test_slot_fit_every_station_export reports the others as skipped)."""
    built = {low: [] for low in L.RL_LOWS}
    for n in _present():
        for low in _assert_agrees(STATION / n, L.slot_fit(_spec(n)), n):
            built[low].append(n)
    # not vacuous: with the opt-outs forced, the exports without a DVL-blend / obs-guard constant to match build
    # outright whatever the dvl: block says (the axes ones on rl, the oldest pwm ones on both pwm LOWs)
    assert {"traj5_s2", "traj10_s1", "traj12_s2"} <= set(built["rl"])
    assert {"pwm11k0.2_s0", "pwm11att7_s0", "pwm11att9X_s0"} <= set(built["rl_pwm"])
    assert {"pwm11att7_s0", "pwm11att9X_s0"} <= set(built["rl_pwm_7d"])
    assert built["rl_7d"] == []


def _mut(s: dict, path: str, value) -> dict:
    s = copy.deepcopy(s)
    *head, last = path.split(".")
    d = s
    for k in head:
        d = d[k]
    if value is _DEL:
        del d[last]
    else:
        d[last] = value
    return s


_DEL = object()
_ALL = {low: True for low in L.RL_LOWS}


@pytest.mark.parametrize("base, path, value, refused", [
    # pwm11att9X_s0: pwm, attitude-trained, no velocity_source / obs_guard / obs_integral block — it builds whatever
    # the dvl: block says, so every slot slot_fit still offers must BUILD outright
    ("pwm11att9X_s0", "attitude_reference.trained", False, {"rl_pwm_7d": _LEVEL}),
    ("pwm11att9X_s0", "attitude_reference.rate_max_rad_s", _DEL,
     {"rl_pwm_7d": "attitude_reference envelope missing (roll/pitch/rate max)"}),
    ("pwm11att9X_s0", "attitude_reference", ["trained"], {"rl_pwm_7d": "attitude_reference is not an object"}),
    ("pwm11att9X_s0", "output.kind", "wrench", {"rl_pwm": "no output.pwm_model (kind pwm) — re-export",
                                                "rl_pwm_7d": "no output.pwm_model (kind pwm) — re-export"}),
    ("pwm11att9X_s0", "output.pwm_model", _DEL, {"rl_pwm": "no output.pwm_model (kind pwm) — re-export",
                                                 "rl_pwm_7d": "no output.pwm_model (kind pwm) — re-export"}),
    ("pwm11att9X_s0", "obs_guard", "yes", _ALL),
    ("pwm11att9X_s0", "obs_guard", {"trained": "true"}, _ALL),
    ("pwm11att9X_s0", "velocity_source", {"trained": 1}, _ALL),
    ("pwm11att9X_s0", "obs_integral", [], _ALL),
    ("pwm11att9X_s0", "obs_integral", {"enabled": "yes"}, _ALL),
    ("pwm11att9X_s0", "obs_goal_frame", "true", _ALL),
    ("pwm11att9X_s0", "history_len", _DEL, _ALL),
    ("pwm11att9X_s0", "action_mode", "wrench6", _ALL),
    # traj12_s2: axes, nothing trained beyond the plain frame
    ("traj12_s2", "obs_guard", {"trained": True, "vel_slew_mps2": 2.0, "jump_m": 0.15}, {"rl": True}),
    ("traj12_s2", "num_actions", "four", _ALL),
])
def test_slot_fit_agrees_with_the_build_on_defects(tmp_path, base, path, value, refused):
    """Synthetic defects of real exports: slot_fit refuses exactly where expected — and the follower build refuses
    there too, while every slot slot_fit still offers builds (or is refused only for a config reason)."""
    s = _mut(_spec(base), path, value)
    fit = L.slot_fit(s)
    want = dict(L.slot_fit(_spec(base)))
    for low, why in refused.items():
        want[low] = why
    for low, why in fit.items():
        if want[low] is True:
            assert why is not None, (low, fit)
        else:
            assert why == want[low], (low, fit)
    d = _write_export(tmp_path / "e", s, npz_from=STATION / base / "policy_weights.npz")
    built = _assert_agrees(d, fit, f"{base} {path}={value!r}")
    assert built == {low for low, why in fit.items() if why is None}


@pytest.mark.parametrize("over, gv, refused", [
    ({}, True, None),
    ({"attitude_reference.trained": False}, True, _LEVEL),
    ({"attitude_reference.roll_max_rad": _DEL}, True, "attitude_reference envelope missing (roll/pitch/rate max)"),
    ({"attitude_reference.pitch_max_rad": 0.0}, True, "attitude_reference envelope missing (roll/pitch/rate max)"),
    ({"obs_guard": {"trained": True, "vel_slew_mps2": 2.0, "jump_m": 0.15}}, True,
     "trained with the pwm obs guard — pwm LOWs only"),
    ({}, False, "no goal feed-forward — the 7D layout needs it"),
])
def test_slot_fit_rl_7d_agrees_with_the_build(tmp_path, over, gv, refused):
    """rl_7d (no station export flies it): an attitude-trained axes6 export fits RL_7D only — level-trained, without
    the trained envelope, trained behind the pwm guard, or without the goal feed-forward it does not."""
    s = _axes6_spec(gv=gv)
    for path, value in over.items():
        s = _mut(s, path, value)
    fit = L.slot_fit(s)
    assert fit == {"rl": _AXES6_ONLY, "rl_pwm": _AXES6_ONLY, "rl_7d": refused, "rl_pwm_7d": _AXES6_ONLY}
    d = _write_export(tmp_path / "a6", s, npz_arrays=_mlp_npz(int(s["obs_dim"]), 6))
    built = _assert_agrees(d, fit, f"axes6 {over} gv={gv}")
    assert built == ({"rl_7d"} if refused is None else set())


def test_slot_fit_never_raises():
    for bad in (None, [], "x", 3):
        assert L.slot_fit(bad) == {low: "obs_spec.json is not a JSON object" for low in L.RL_LOWS}
    fit = L.slot_fit({})
    assert set(fit) == set(L.RL_LOWS) and all(fit.values())
    fit = L.slot_fit({"history_len": 4, "frame_dim": 25, "num_actions": 4, "action_mode": None})
    assert fit == {low: "action_mode None is not flown by the station" for low in L.RL_LOWS}


# ----------------------------------------------------------------------------- entries / scan / dedupe / candidates
def test_station_entries():
    ents = {e.name: e for e in L.scan() if e.where == "station"}
    assert set(ents) == set(_present())                         # = the folder (test_every_station_export_has_a_row)
    for n, e in ents.items():
        assert e.error == "" and e.path == str(STATION_ABS / n) and e.rel == f"rov_gui/control/rl_policies/{n}"
        assert (e.obs_sha1, e.npz_sha1) == L.export_hashes(STATION / n)
        assert e.id == L.export_id(e.obs_sha1, e.npz_sha1)
        assert e.fit == EXPECTED_FIT[n]
        assert (e.no_fly is not None) == (n in EXPECTED_NO_FLY), n
    e = ents["pwm16N_s3"]
    assert (e.action_mode, e.output_kind, e.att_trained, e.cap_trained, e.dr_preset, e.iteration) == \
        ("pwm", "pwm", True, 0.5, "pwm15", 2399)
    assert (e.trained, e.vel_trained, e.integral, e.source) == ("2026-10-19 03:25", True, True, "")
    assert e.checkpoint == "logs/rsl_rl/rov_posehold/2026-10-19_03-25-36_pwm16N_s3/model_2399.pt"
    e = ents["traj5_s2"]                                        # no action_mode key: axes, as HwRl reads it
    assert (e.action_mode, e.output_kind, e.cap_trained, e.att_trained, e.trained) == \
        ("axes", "wrench", None, False, "2026-09-21 16:53")
    assert ents["pwm10_s1"].cap_trained == 1.0 and ents["pwm11k0.2_s0"].att_trained is False
    hash(e)                                                      # frozen + fit left out of ==/hash
    assert L.dedupe(L.scan()) == L.dedupe(L.scan())


def test_scan_dedupe_candidates(tmp_path):
    st, lib = tmp_path / "station", tmp_path / "library"
    _copy_station("pwm16N_s3", st / "a")
    _copy_station("pwm15N_s0", st / ".tmp-b-1234")              # the sync tool's temp dir: skipped
    (st / "notes").mkdir()                                      # no obs_spec.json: not an export, ignored
    (st / "notes" / "policy_weights.npz").write_bytes(b"x")
    (st / "README.md").write_text("x")                          # a plain file at the root: ignored
    (st / "broken").mkdir()
    (st / "broken" / "obs_spec.json").write_text("{", encoding="utf-8")
    shutil.copyfile(STATION / "traj12_s2" / "policy_weights.npz", st / "broken" / "policy_weights.npz")
    _copy_station("pwm16N_s3", lib / "a_copy")                  # same bytes as station 'a': deduped to the station's
    _copy_station("pwm15N_s0", lib / "pwm15N_s0")
    _copy_station("pwm11att14R_s1", lib / "pwm11att14R_s1")    # git-tracked (2026-11-02: was the untracked ..Rw_s1)
    _copy_station("traj12_s2", lib / "traj12_s2")
    (lib / "nonpz").mkdir()
    shutil.copyfile(STATION / "traj12_s2" / "obs_spec.json", lib / "nonpz" / "obs_spec.json")
    # same trained stamp as 'a' (its obs_spec) but other weights: ties are broken by name
    _write_export(lib / "aa_tie", _spec("pwm16N_s3"), npz_from=STATION / "pwm15N_s0" / "policy_weights.npz")
    # no run stamp anywhere: sorts last
    s = _spec("pwm14B_s0")
    s["checkpoint"] = "model_2399.pt"
    _write_export(lib / "zz_nostamp", s, npz_from=STATION / "pwm14B_s0" / "policy_weights.npz")

    ents = L.scan([("station", st), ("library", lib)])
    assert [(e.where, e.name) for e in ents] == [
        ("station", "a"), ("station", "broken"),
        ("library", "a_copy"), ("library", "aa_tie"), ("library", "nonpz"), ("library", "pwm11att14R_s1"),
        ("library", "pwm15N_s0"), ("library", "traj12_s2"), ("library", "zz_nostamp")]
    by = {e.name: e for e in ents}
    assert "not valid JSON" in by["broken"].error and by["broken"].obs_sha1 and by["broken"].id
    assert by["nonpz"].error == "no policy_weights.npz" and by["nonpz"].npz_sha1 == ""
    for n in ("broken", "nonpz"):
        assert all(by[n].fit[low] for low in L.RL_LOWS)          # an error entry fits nowhere
    assert by["a"].id == by["a_copy"].id
    assert by["zz_nostamp"].trained == "" and by["zz_nostamp"].iteration == 2399

    dd = L.dedupe(ents)
    assert [e.name for e in dd] == ["a", "broken", "aa_tie", "nonpz", "pwm11att14R_s1", "pwm15N_s0", "traj12_s2",
                                    "zz_nostamp"]
    assert next(e for e in dd if e.id == by["a"].id).where == "station"
    # errors are all kept, even two with the same (empty) id
    assert len(L.dedupe([by["nonpz"], by["nonpz"], by["broken"]])) == 3

    c = L.candidates(dd, "rl_pwm_7d")                           # newest trained first (across both roots), ties by name
    assert [(e.name, e.trained) for e in c] == [
        ("a", "2026-10-19 03:25"), ("aa_tie", "2026-10-19 03:25"), ("pwm15N_s0", "2026-10-18 22:54"),
        ("pwm11att14R_s1", "2026-10-16 06:00"), ("zz_nostamp", "")]
    assert [e.name for e in L.candidates(dd, "rl")] == ["traj12_s2"]
    assert L.candidates(dd, "rl_7d") == []
    allc = L.candidates(dd, "rl_7d", include_refused=True)
    assert {e.name for e in allc} == {e.name for e in dd}       # refused, errors and all
    assert [e.trained for e in allc] == sorted((e.trained for e in allc), reverse=True)
    assert {"broken", "nonpz"} <= {e.name for e in L.candidates(dd, "rl", include_refused=True)}
    with pytest.raises(ValueError):
        L.candidates(dd, "mpc")
    # an Entry without a verdict for the slot is NOT offered (fail-closed)
    bare = L.Entry(name="x", path="/x", rel="/x", where="other", id="abc", trained="2099-01-01 00:00")
    assert L.candidates([bare], "rl") == []


def test_refusal_reason_order_and_the_iteration_floor(tmp_path):
    """refusal_reason (2026-11-02) — the ONE rule of candidates(), the picker dialog and `./c3 rl-policies list` — first
    hit wins: unreadable -> flight ban -> the iteration floor -> the slot pre-filter (a missing verdict refuses). A
    hand-copied SMOKE export (complete obs_spec, rsl_rl_iteration 1-2) passes slot_fit and the follower build, so the
    floor alone keeps it out of the picker: refused for every LOW, listed only with include_refused."""
    lib = tmp_path / "lib"
    npz = STATION / "pwm11att9X_s0" / "policy_weights.npz"        # builds outright with the opt-outs forced
    base = _spec("pwm11att9X_s0")
    noit = {k: v for k, v in base.items() if k != "rsl_rl_iteration"}
    _write_export(lib / "smoke_copy", {**base, "rsl_rl_iteration": 2}, npz_from=npz)
    _write_export(lib / "at_floor", {**base, "rsl_rl_iteration": L.MIN_ITERATION}, npz_from=npz)
    # the floor reads Entry.iteration, fallbacks included: no rsl_rl_iteration, the checkpoint's model_1 ...
    _write_export(lib / "model_1", {**noit, "checkpoint": base["checkpoint"].replace("model_1799", "model_1")},
                  npz_from=npz)
    # ... while an UNKNOWN iteration is left to the slot rule (and the build)
    _write_export(lib / "no_iter", {**noit, "checkpoint": "Policies/x/policy.pt"}, npz_from=npz)
    # banned AND smoke-length: the ban speaks
    _write_export(lib / "pwm10_s2", {**_spec("pwm10_s1"), "rsl_rl_iteration": 1},
                  npz_from=STATION / "pwm10_s1" / "policy_weights.npz")
    _write_export(lib / "broken", {**base, "rsl_rl_iteration": 1})                    # no npz: unreadable
    ents = {e.name: e for e in L.scan([("library", lib)])}
    floor = f"smoke-length export (2 iterations < {L.MIN_ITERATION})"
    # the slot pre-filter AND the follower build would both take it — the floor is the only thing that refuses it
    assert ents["smoke_copy"].fit["rl_pwm_7d"] is None and ents["smoke_copy"].fit["rl_pwm"] is None
    assert _dry_build("rl_pwm_7d", lib / "smoke_copy") is None and _dry_build("rl_pwm", lib / "smoke_copy") is None
    assert {low: L.refusal_reason(ents["smoke_copy"], low) for low in L.RL_LOWS} == {low: floor for low in L.RL_LOWS}
    assert ents["model_1"].iteration == 1
    assert L.refusal_reason(ents["model_1"], "rl_pwm") == f"smoke-length export (1 iterations < {L.MIN_ITERATION})"
    assert ents["no_iter"].iteration is None and L.refusal_reason(ents["no_iter"], "rl_pwm_7d") is None
    assert L.refusal_reason(ents["at_floor"], "rl_pwm_7d") is None
    assert L.refusal_reason(ents["at_floor"], "rl") == "pwm export — RL_PWM / RL_PWM_7D only"
    assert L.refusal_reason(ents["pwm10_s2"], "rl_pwm") == L.NO_FLY["pwm10_s2"]
    assert L.refusal_reason(ents["broken"], "rl_pwm") == "unusable export — no policy_weights.npz"
    every = list(ents.values())
    assert [e.name for e in L.candidates(every, "rl_pwm_7d")] == ["at_floor", "no_iter"]
    allc = L.candidates(every, "rl_pwm_7d", include_refused=True)
    assert {e.name for e in allc} == set(ents) and allc[-1].name == "no_iter"        # unknown 'trained' sorts last
    # bare entries: the floor needs no file; a missing verdict / an unknown LOW refuses; a refusal is never ""
    fits = {low: None for low in L.RL_LOWS}
    bare = dict(name="x", path="/x", rel="/x", where="other", id="abc", trained="2099-01-01 00:00")
    assert L.refusal_reason(L.Entry(**bare, iteration=L.MIN_ITERATION - 1, fit=fits), "rl") == \
        f"smoke-length export ({L.MIN_ITERATION - 1} iterations < {L.MIN_ITERATION})"
    assert L.refusal_reason(L.Entry(**bare, fit=fits), "rl") is None
    assert L.refusal_reason(L.Entry(**bare), "rl") == "not evaluated for this LOW"
    assert L.refusal_reason(L.Entry(**bare, fit=fits), "mpc") == "not evaluated for this LOW"
    assert L.refusal_reason(L.Entry(**bare, fit=fits, no_fly=""), "rl") == "flight ban"


def test_the_iteration_floor_changes_no_station_export():
    """MIN_ITERATION (2026-11-02) refuses no station export: each trained >= the floor, and refusal_reason /
    candidates() give exactly the rule from before the floor (no error, no ban, fit[low] None) for every LOW."""
    ents = L.dedupe(L.scan([L.LIBRARY_ROOTS[0]]))
    assert {e.name for e in ents} == set(_present())
    for e in ents:
        assert e.iteration is not None and e.iteration >= L.MIN_ITERATION, (e.name, e.iteration)
    for low in L.RL_LOWS:
        before = [e for e in ents if not e.error and e.no_fly is None and e.fit.get(low, "not evaluated") is None]
        before.sort(key=lambda e: e.name)
        before.sort(key=lambda e: e.trained, reverse=True)
        assert [e.name for e in L.candidates(ents, low)] == [e.name for e in before], low
        for e in ents:
            why = L.refusal_reason(e, low)
            assert (why is None) == (e.name in {x.name for x in before}), (e.name, low, why)
            if why is not None and e.no_fly is None:
                assert why == e.fit[low], (e.name, low)                                # the slot's own text


# ----------------------------------------------------------------------------- the transport cap (cap_ctx, 2026-11-02)
#: the LOW row's cap ▾ menu — a copy of workers.RL_PWM_CAP_CHOICES (test_rl_pwm_cap.py pins that tuple to these values)
_CAP_MENU = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
_NO_CAP = "no LOW-row cap flies it here"


def _cap_ctx(cap, ceiling=None, allow_pwm=False, allow_7d=True) -> dict:
    """MpcStatus.rl_pwm_cap as MpcWorker._rl_pwm_cap_status publishes it (the LOW row's cap ▾, 2026-11-02):
    refusal_reason reads "cap", "ceiling" and "allow_below"; the rest rides along as on the wire. The default opt-outs
    are config/hw_mpc.yaml's (rl_pwm.allow_caps_below_trained false, rl_pwm_7d.allow_caps_below_trained true)."""
    return {"cap": cap, "source": "hw_mpc.yaml", "at": None, "choices": list(_CAP_MENU), "ceiling": ceiling,
            "sink_cap": None, "allow_below": {"rl_pwm": allow_pwm, "rl_pwm_7d": allow_7d}, "state": "ok", "note": ""}


def _bare(**kw) -> L.Entry:
    """A readable, unbanned entry every LOW's pre-filter accepts — only what the test sets differs."""
    return L.Entry(name="x", path="/x", rel="/x", where="other", id="abc", fit={low: None for low in L.RL_LOWS}, **kw)


def test_cap_ctx_none_or_empty_is_exactly_the_rule_of_before():
    """refusal_reason(entry, low, cap_ctx) — the cap SPEC's C8. cap_ctx None / {} (candidates(), `./c3 rl-policies
    list`) is EXACTLY refusal_reason(entry, low), for every station export and LOW; so is a ctx without a finite cap
    (a worker with no config publishes {}), any ctx for an axes LOW, and an entry whose trained cap is unknown."""
    ents = {e.name: e for e in L.scan() if e.where == "station"}
    junk = ({"cap": None}, {"cap": "0.25"}, {"cap": float("nan")}, {"cap": float("inf")}, {"cap": True},
            {"ceiling": 0.25, "allow_below": {"rl_pwm": False, "rl_pwm_7d": False}})
    tight = _cap_ctx(0.1, ceiling=0.1, allow_pwm=False, allow_7d=False)
    for e in ents.values():
        for low in L.RL_LOWS:
            base = L.refusal_reason(e, low)
            assert L.refusal_reason(e, low, None) == base and L.refusal_reason(e, low, {}) == base, (e.name, low)
            for ctx in junk:
                assert L.refusal_reason(e, low, ctx) == base, (e.name, low, ctx)
            if low in ("rl", "rl_7d"):
                assert L.refusal_reason(e, low, tight) == base, (e.name, low)
    unknown = _bare()
    assert unknown.cap_trained is None and L.refusal_reason(unknown, "rl_pwm", tight) is None, "nothing to judge"
    # pwm LOWs only — even for an entry with a trained cap that an axes LOW's pre-filter took (no station export is
    # one: an axes export has no pwm_cap_trained, so this is the only way the rule's own LOW guard shows)
    capped = _bare(cap_trained=0.5)
    assert L.refusal_reason(capped, "rl_pwm", tight) is not None
    assert L.refusal_reason(capped, "rl", tight) is None and L.refusal_reason(capped, "rl_7d", tight) is None


def test_cap_ctx_speaks_only_after_every_earlier_check_passed():
    """The cap is step 5: an unreadable / banned / smoke-length / ill-fitting export keeps its own reason under a ctx
    that would refuse it too (every banned station export is trained at 1.0 — (a) at any cap below it)."""
    ents = {e.name: e for e in L.scan() if e.where == "station"}
    tight = _cap_ctx(0.1, ceiling=0.25, allow_pwm=False, allow_7d=False)
    for name in EXPECTED_NO_FLY:
        for low in ("rl_pwm", "rl_pwm_7d"):
            assert L.refusal_reason(ents[name], low, tight) == L.refusal_reason(ents[name], low) == \
                str(ents[name].no_fly), (name, low)
    assert L.refusal_reason(ents["pwm11k0.2_s0"], "rl_pwm_7d", tight) == _LEVEL
    assert L.refusal_reason(ents["traj12_s2"], "rl_pwm", tight) == _AXES_ONLY
    assert L.refusal_reason(_bare(cap_trained=0.5, error="no policy_weights.npz"), "rl_pwm", tight) == \
        "unusable export — no policy_weights.npz"
    assert L.refusal_reason(_bare(cap_trained=0.5, iteration=1), "rl_pwm", tight) == \
        f"smoke-length export (1 iterations < {L.MIN_ITERATION})"


def test_cap_ctx_a_a_cap_below_trained_where_the_low_refuses_it():
    """(a) the export's trained cap above the transport cap, and the LOW's allow_caps_below_trained false (HwRlPwm /
    HwRlPwm7d refuse that build): "trained at cap T > the transport cap C and <LOW> refuses caps below trained — set
    the LOW row's cap to T". As shipped (0.25; rl_pwm false, rl_pwm_7d true) every 0.5-trained export is refused for
    RL_PWM and offered to RL_PWM_7D."""
    ents = {e.name: e for e in L.scan() if e.where == "station"}
    lvl, att = ents["pwm11k0.2_s0"], ents["pwm11att9X_s0"]
    shipped = _cap_ctx(0.25)
    for e in (lvl, att):
        assert L.refusal_reason(e, "rl_pwm", shipped) == ("trained at cap 0.5 > the transport cap 0.25 and RL_PWM "
                                                          "refuses caps below trained — set the LOW row's cap to "
                                                          "0.5"), e.name
    assert L.refusal_reason(att, "rl_pwm_7d", shipped) is None, "its own opt-out"
    assert L.refusal_reason(att, "rl_pwm_7d", _cap_ctx(0.25, allow_7d=False)) == (
        "trained at cap 0.5 > the transport cap 0.25 and RL_PWM_7D refuses caps below trained — set the LOW row's cap "
        "to 0.5")
    assert L.refusal_reason(lvl, "rl_pwm", _cap_ctx(0.25, allow_pwm=True)) is None
    # at / above the trained cap nothing (the follower flies min(cap, trained)); the boundary is 1e-9
    for cap in (0.5, 0.5 - 1e-10, 0.75, 1.0):
        assert L.refusal_reason(lvl, "rl_pwm", _cap_ctx(cap)) is None, cap
    assert L.refusal_reason(lvl, "rl_pwm", _cap_ctx(0.5 - 1e-6)) is not None
    # an opt-out the ctx does not carry is no opt-out (fail-closed)
    for allow in (None, {}, {"rl_pwm": True}, "rl_pwm_7d"):
        assert L.refusal_reason(att, "rl_pwm_7d", {**shipped, "allow_below": allow}) is not None, allow
    ctx = dict(shipped)
    del ctx["allow_below"]
    assert L.refusal_reason(att, "rl_pwm_7d", ctx) is not None


def test_cap_ctx_b_the_sinks_window_differs_from_the_followers():
    """(b) the sink's ceiling known (its gate export's trained cap) and min(cap, ceiling) != min(cap, trained): the
    sink would enforce another window than the follower flies (workers._sink_gate_compare refuses that pick)."""
    t25 = _bare(cap_trained=0.25)                              # trained BELOW the sink export's 0.5
    for low in ("rl_pwm", "rl_pwm_7d"):
        assert L.refusal_reason(t25, low, _cap_ctx(0.5, ceiling=0.5)) == \
            "flies cap 0.25 but the thruster sink's window is 0.5 — set the LOW row's cap to 0.25", low
        assert L.refusal_reason(t25, low, _cap_ctx(0.25, ceiling=0.5)) is None, "the remedy holds"
        assert L.refusal_reason(t25, low, _cap_ctx(0.5)) is None, "no ceiling known: no (b)"
    for cap in (0.2, 0.1):                                     # below its 0.25 the windows agree again — (a) decides
        assert L.refusal_reason(t25, "rl_pwm_7d", _cap_ctx(cap, ceiling=0.5)) is None, cap
        assert L.refusal_reason(t25, "rl_pwm", _cap_ctx(cap, ceiling=0.5)).startswith("trained at cap 0.25 > "), cap
    # equal trained caps — every unbanned station pwm export and the shipped sink export are trained at 0.5: never (b)
    att = {e.name: e for e in L.scan() if e.where == "station"}["pwm11att9X_s0"]
    for cap in _CAP_MENU:
        if cap <= 0.5:
            assert L.refusal_reason(att, "rl_pwm_7d", _cap_ctx(cap, ceiling=0.5)) is None, cap
    # trained ABOVE the ceiling (a sink export trained lower than this one). The SPEC's literal remedy "set the LOW
    # row's cap to {T}" would name a cap set_rl_pwm_cap refuses (its step 7: above the sink's ceiling), so the remedy
    # is clipped to the ceiling — where both windows agree for a LOW that allows caps below trained ...
    assert L.refusal_reason(att, "rl_pwm_7d", _cap_ctx(0.5, ceiling=0.25)) == \
        "flies cap 0.5 but the thruster sink's window is 0.25 — set the LOW row's cap to 0.25"
    assert L.refusal_reason(att, "rl_pwm_7d", _cap_ctx(0.25, ceiling=0.25)) is None, "the remedy holds"
    # ... and a LOW that refuses caps below trained has NO cap on the row (below T it is (a), at or above the ceiling
    # (b)) — said so, instead of a remedy the next look refuses
    for cap in (0.1, 0.25, 0.5, 1.0):
        assert L.refusal_reason(att, "rl_pwm", _cap_ctx(cap, ceiling=0.25)) == (
            f"trained at cap 0.5, above the thruster sink's window 0.25, and RL_PWM refuses caps below trained — "
            f"{_NO_CAP}"), cap


def test_cap_ctx_every_remedy_it_names_offers_the_export():
    """No remedy leads to the next refusal: whatever (a) / (b) tell the operator — "set the LOW row's cap to X" — X is
    a cap set_rl_pwm_cap accepts (in (0, 1], not above a known ceiling) at which the same rule OFFERS the export; and
    where the text says no cap flies it, none on the menu does. Over the unbanned station pwm exports and synthetic
    trained caps, both pwm LOWs, both opt-outs, every ceiling and every menu cap the worker would accept."""
    import re

    ents = [e for e in L.scan() if e.where == "station" and e.cap_trained is not None and e.no_fly is None]
    ents += [_bare(cap_trained=t) for t in (0.1, 0.25, 0.3, 0.5, 0.75, 1.0)]
    named = 0
    for e in ents:
        for low in ("rl_pwm", "rl_pwm_7d"):
            if L.refusal_reason(e, low) is not None:
                continue                                       # refused before the cap: not this rule's business
            for allow in (False, True):
                for ceiling in (None, 0.25, 0.5, 1.0):
                    ok_caps = [c for c in _CAP_MENU if ceiling is None or c <= ceiling + 1e-9]

                    def why(c, e=e, low=low, allow=allow, ceiling=ceiling):
                        return L.refusal_reason(e, low, _cap_ctx(c, ceiling, allow_pwm=allow, allow_7d=allow))

                    for cap in ok_caps:
                        w = why(cap)
                        if w is None:
                            continue
                        m = re.search(r"set the LOW row's cap to ([0-9.]+)$", w)
                        if m is None:
                            assert w.endswith(_NO_CAP), w
                            assert all(why(c) is not None for c in ok_caps), (e.name, low, allow, ceiling, w)
                            continue
                        x = float(m.group(1))
                        assert 0.0 < x <= 1.0 and (ceiling is None or x <= ceiling + 1e-9), (w, ceiling)
                        assert why(x) is None, (e.name, e.cap_trained, low, allow, ceiling, cap, w)
                        named += 1
    assert named > 0


def test_scan_roots(tmp_path):
    assert L.scan([("library", tmp_path / "missing")]) == []
    assert L.scan(tmp_path / "missing") == []
    _copy_station("traj12_s2", tmp_path / "r" / "t")
    (e,) = L.scan(tmp_path / "r")                               # a bare path: "other"
    assert (e.where, e.rel) == ("other", str(tmp_path / "r" / "t"))
    assert {x.where for x in L.scan(str(STATION))} == {"station"}  # a bare path that IS a library root keeps its label
    assert {x.where for x in L.scan("rov_gui/control/rl_policies")} == {"station"}


def test_sidecar_provenance_and_integrity(tmp_path):
    lib = tmp_path / "lib"
    s = _spec("pwm16N_s3")
    s["checkpoint"] = "Policies/pwm_k0_s0/model_1199.pt"          # no run stamp in the checkpoint
    d = _write_export(lib / "pwm_k0_s0", s, npz_from=STATION / "pwm16N_s3" / "policy_weights.npz")
    obs, npz = L.export_hashes(d)
    side = {"schema": 1, "name": "pwm_k0_s0", "source_dir": "/home/bdml/Desktop/RL_controller/Policies/pwm_k0_s0",
            "run_stamp": "2026-10-09T12:49:17", "obs_sha1": obs, "npz_sha1": npz, "id": L.export_id(obs, npz)}
    (d / L.SIDECAR).write_text(json.dumps(side), encoding="utf-8")
    (e,) = L.scan([("library", lib)])
    assert e.error == "" and e.source == side["source_dir"] and e.trained == "2026-10-09 12:49"
    # an entry modified after the sync is never offered (immutability)
    (d / L.SIDECAR).write_text(json.dumps({**side, "npz_sha1": "0" * 40}), encoding="utf-8")
    _bump_mtime(d / L.SIDECAR)
    (e,) = L.scan([("library", lib)])
    assert "policy_weights.npz changed since the sync" in e.error and e.source == side["source_dir"]
    assert L.candidates([e], "rl_pwm_7d") == []
    # an unreadable sidecar is provenance lost, not an error (the files themselves are hashed and checked)
    (d / L.SIDECAR).write_text("{", encoding="utf-8")
    _bump_mtime(d / L.SIDECAR)
    (e,) = L.scan([("library", lib)])
    assert e.error == "" and e.source == "" and e.trained == ""


def test_scan_cache_rereads_a_changed_file(tmp_path):
    root = tmp_path / "lib"
    d = _copy_station("pwm16N_s3", root / "x")
    (e1,) = L.scan([("library", root)])
    (e1b,) = L.scan([("library", root)])
    assert e1b is e1                                            # unchanged: served from the cache
    s = _spec("pwm16N_s3")
    s["dr_preset"] = "pwm15_longer_name"                        # another size
    (d / "obs_spec.json").write_text(json.dumps(s), encoding="utf-8")
    (e2,) = L.scan([("library", root)])
    assert e2.dr_preset == "pwm15_longer_name" and e2.obs_sha1 != e1.obs_sha1 and e2.id != e1.id
    # same size, new bytes: the mtime (bumped explicitly; the filesystem clock is coarse) re-reads it
    s["dr_preset"] = "pwm15_longer_namf"
    (d / "obs_spec.json").write_text(json.dumps(s), encoding="utf-8")
    _bump_mtime(d / "obs_spec.json")
    (e3,) = L.scan([("library", root)])
    assert e3.dr_preset == "pwm15_longer_namf" and e3.obs_sha1 != e2.obs_sha1
    # the npz too (same architecture = same size: the mtime tells)
    shutil.copyfile(STATION / "pwm15N_s0" / "policy_weights.npz", d / "policy_weights.npz")
    _bump_mtime(d / "policy_weights.npz")
    (e4,) = L.scan([("library", root)])
    assert e4.npz_sha1 == L.file_sha1(STATION / "pwm15N_s0" / "policy_weights.npz") and e4.id != e3.id
    # and a file that disappears
    (d / "policy_weights.npz").unlink()
    (e5,) = L.scan([("library", root)])
    assert e5.error == "no policy_weights.npz"
