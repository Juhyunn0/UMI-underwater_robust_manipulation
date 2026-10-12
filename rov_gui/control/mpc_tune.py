#!/usr/bin/env python3
"""mpc_tune.py — MPC_Tuned PROFILES: the follower's tuning, saved to a file
and picked on the panel the way a Diffusion Policy checkpoint is (2026-10-04,
operator request: "mpc_tuned를 선택하면 diffusion policy처럼 튜닝 파라미터를
불러와서 적용").

Before this, what mpc_tuned flew was scattered over hw_mpc.yaml (``mpc_tuned:``,
``policy.q_scale`` / ``along_scale``, ``vehicle_net_buoyancy_n``, ``axis_cap``,
``axis_slew_per_s``) and params.py (Q / QN / R), and the only record of "the
tune that worked" was the run meta of the run that used it. A PROFILE is one
YAML file in ``config/mpc_tuned_profiles/`` holding all of it, in three
sections that differ in what the station DOES with them:

  apply     APPLIED when the profile is picked — runtime only, no rebuild —
            and ONLY while the LOW level is mpc_tuned (mpc_tuned_7d flies the
            same HwDobMpc mode, so it gets it too). mpc / dobmpc /
            dobmpc_tuned / pid keep hw_mpc.yaml whatever is picked here.
              Q, QN, R            the isotropic baseline the split is built on
              path_cost           the along/cross split (path_cost.TUNE_KEYS)
              policy              q_scale / along_scale while a DP plan is installed
              vehicle_net_buoyancy_n   the constant heave trim (B - W, + floats)
              axis_cap, axis_slew_per_s   the output stage
            EVERY key is required: a missing one would fall back to
            hw_mpc.yaml silently, and a run would fly something other than
            the profile its meta names.
  compare   NOT applied — the solver and vehicle model the tune was made on
            (rov_model, ctrl_hz, N, dt_s, u_max, plant damping, axis_gain).
            The damping is baked into the generated solver, which mpc and
            dobmpc share, so a profile cannot carry it in at run time. A
            difference from the running station is a WARNING on the panel
            and in the run meta (``controller.mpc_tune.warnings``), never a
            silent pass.
  record    context only, never read back: the prediction model, the path
            shaping, the hw_mpc.yaml blocks as they stood when it was saved.

    python -m rov_gui.tools.mpc_tune save NAME --description "..."   # snapshot
    python -m rov_gui.tools.mpc_tune list | show NAME

Pure Python + numpy + yaml: nothing here imports acados.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Where profiles live. In the repo (git-tracked, a few kB each), not under
#: data/: a profile is configuration, and a run meta that names one must be
#: able to point at a file that is still there.
PROFILE_DIR = REPO_ROOT / "config" / "mpc_tuned_profiles"
SCHEMA = 1

#: The names that mean "no profile — fly hw_mpc.yaml's own blocks" (the
#: behaviour before 2026-10-04). CLI ``--mpc-tune none`` / YAML null.
NO_PROFILE = ("", "none", "null", "hw_mpc.yaml")

TOP_KEYS = ("schema", "name", "created", "description", "source",
            "apply", "compare", "record")
APPLY_KEYS = ("Q", "QN", "R", "path_cost", "policy",
              "vehicle_net_buoyancy_n", "axis_cap", "axis_slew_per_s")
POLICY_KEYS = ("q_scale", "along_scale")
COMPARE_KEYS = ("rov_model", "ctrl_hz", "N", "dt_s", "u_max",
                "linear_damping", "quadratic_damping", "axis_gain")
#: x = [x y z, phi theta psi, u v w, p q r]; u = [X Y Z K M N] (heavy, rank-6)
NX, NU = 12, 6


@dataclass
class MpcTuneProfile:
    """One loaded, validated profile. ``apply`` is normalised (floats, the
    full path_cost block); ``sha1`` is of the file's bytes, which is what the
    run meta pins (a profile edited in place after a run is a different
    profile under the same name)."""
    name: str
    path: str
    sha1: str
    apply: dict
    compare: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)
    created: str = ""
    description: str = ""
    source: dict = field(default_factory=dict)

    def meta(self) -> dict:
        return {"profile": self.name, "path": self.path, "sha1": self.sha1,
                "created": self.created, "description": self.description,
                "apply": _plain(self.apply), "compare": _plain(self.compare),
                "source": _plain(self.source)}


# --------------------------------------------------------------------- paths
def is_no_profile(name) -> bool:
    return name is None or str(name).strip().lower() in NO_PROFILE


def resolve_path(name_or_path) -> Path:
    """A NAME (``20261004_default``) is looked up in PROFILE_DIR; anything
    that already names an existing file (relative to the cwd or the repo) is
    taken as it is."""
    s = str(name_or_path).strip()
    p = Path(s).expanduser()
    for cand in (p, REPO_ROOT / p):
        if cand.is_file():
            return cand.resolve()
    stem = s[:-5] if s.endswith(".yaml") else s
    return (PROFILE_DIR / f"{stem}.yaml").resolve()


def list_profiles(directory=None) -> list[dict]:
    """``[{name, path, description, created}]`` sorted by name, for the CLI
    (the panel uses a file dialog on the same directory). A file that does
    not load is listed with its error instead of being hidden."""
    d = Path(directory) if directory is not None else PROFILE_DIR
    out = []
    for p in sorted(d.glob("*.yaml")):
        try:
            prof = load_profile(p)
            out.append({"name": prof.name, "path": str(p),
                        "description": prof.description,
                        "created": prof.created, "error": ""})
        except Exception as e:                                   # noqa: BLE001
            out.append({"name": p.stem, "path": str(p), "description": "",
                        "created": "", "error": f"{type(e).__name__}: {e}"})
    return out


# ---------------------------------------------------------------- validation
def _vec(v, n: int, what: str, *, positive: bool = False) -> list:
    a = np.asarray(v, float).ravel()
    if a.size != n or not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError(f"{what} must be {n} finite non-negative numbers, got {v!r}")
    if positive and np.any(a <= 0.0):
        raise ValueError(f"{what} must be strictly positive, got {v!r}")
    return [float(x) for x in a]


def resolve_apply(raw: dict) -> dict:
    """Validate and normalise an ``apply:`` block. Unknown and MISSING keys
    raise (see the module docstring for why a missing key is an error)."""
    from .geometry import check_vehicle_net_buoyancy
    from .path_cost import TUNE_KEYS, resolve_tune

    if not isinstance(raw, dict):
        raise ValueError(f"apply: must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - set(APPLY_KEYS)
    if unknown:
        raise ValueError(f"apply: unknown key(s) {sorted(unknown)}; "
                         f"known {list(APPLY_KEYS)}")
    missing = [k for k in APPLY_KEYS if k not in raw]
    if missing:
        raise ValueError(f"apply: missing key(s) {missing} — a profile is "
                         f"complete or it is not loaded")
    out = {"Q": _vec(raw["Q"], NX, "apply.Q"),
           "QN": _vec(raw["QN"], NX, "apply.QN"),
           # R > 0: a zero input weight makes the LS cost singular in u
           "R": _vec(raw["R"], NU, "apply.R", positive=True)}
    for k in ("Q", "QN"):
        # The POSE rows [x y z phi theta psi] must hold something: a zero
        # x/y weight is no station hold at all, and q_along = 0 divides by
        # zero in PathFrameWeights.meta() at CSV open (safety review
        # 2026-10-04). The rate rows may be 0.
        if any(v <= 0.0 for v in out[k][:6]):
            raise ValueError(f"apply.{k}[0:6] (x y z phi theta psi) must be "
                             f"> 0, got {out[k][:6]}")
    pc = dict(raw["path_cost"] or {})
    missing = [k for k in TUNE_KEYS if k not in pc]
    if missing:
        # resolve_tune would fill these from DEFAULT_TUNE — a value nobody
        # wrote into this profile (review 2026-10-04)
        raise ValueError(f"apply.path_cost: missing key(s) {missing}")
    out["path_cost"] = resolve_tune(pc)
    pol = raw["policy"] or {}
    unknown = set(pol) - set(POLICY_KEYS)
    missing = [k for k in POLICY_KEYS if k not in pol]
    if unknown or missing:
        raise ValueError(f"apply.policy: unknown {sorted(unknown)} / missing "
                         f"{missing}; keys {list(POLICY_KEYS)}")
    q = float(pol["q_scale"])
    if not (math.isfinite(q) and q > 0.0):
        raise ValueError(f"apply.policy.q_scale must be > 0, got {pol['q_scale']!r}")
    a = pol["along_scale"]
    if a is not None:
        a = float(a)
        if not (math.isfinite(a) and a > 0.0):
            raise ValueError(f"apply.policy.along_scale must be null or > 0, "
                             f"got {pol['along_scale']!r}")
    out["policy"] = {"q_scale": q, "along_scale": a}
    out["vehicle_net_buoyancy_n"] = check_vehicle_net_buoyancy(
        raw["vehicle_net_buoyancy_n"])
    cap = float(raw["axis_cap"])
    if not (math.isfinite(cap) and 0.0 < cap <= 1.0):
        raise ValueError(f"apply.axis_cap must be in (0, 1], got {raw['axis_cap']!r}")
    out["axis_cap"] = cap
    slew = float(raw["axis_slew_per_s"])
    if not (math.isfinite(slew) and slew > 0.0):
        # 0 = NO slew limit (allocation.slew_axes passes through), i.e. a
        # profile could switch off the station's only anti-chatter device
        # (hw_mpc.yaml axis_slew_per_s). Tighten-only, like the cap.
        raise ValueError(f"apply.axis_slew_per_s must be > 0, got "
                         f"{raw['axis_slew_per_s']!r}")
    out["axis_slew_per_s"] = slew
    return out


def resolve_compare(raw: dict | None) -> dict:
    raw = dict(raw or {})
    unknown = set(raw) - set(COMPARE_KEYS)
    if unknown:
        raise ValueError(f"compare: unknown key(s) {sorted(unknown)}; "
                         f"known {list(COMPARE_KEYS)}")
    return raw


def load_profile(name_or_path) -> MpcTuneProfile:
    """Read + validate one profile. Raises FileNotFoundError / ValueError
    with a sentence the panel can show as it is."""
    import yaml

    path = resolve_path(name_or_path)
    if not path.is_file():
        raise FileNotFoundError(f"no MPC_Tuned profile {str(name_or_path)!r} "
                                f"({path})")
    data = path.read_bytes()
    raw = yaml.safe_load(data.decode("utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: not a mapping")
    unknown = set(raw) - set(TOP_KEYS)
    if unknown:
        raise ValueError(f"{path.name}: unknown top-level key(s) "
                         f"{sorted(unknown)}; known {list(TOP_KEYS)}")
    schema = int(raw.get("schema", 0) or 0)
    if schema != SCHEMA:
        raise ValueError(f"{path.name}: schema {schema}, this station reads "
                         f"{SCHEMA}")
    if "apply" not in raw:
        raise ValueError(f"{path.name}: no apply: block")
    try:
        apply = resolve_apply(raw["apply"])
        compare = resolve_compare(raw.get("compare"))
    except ValueError as e:
        raise ValueError(f"{path.name}: {e}") from e
    name = str(raw.get("name") or path.stem)
    if name != path.stem:
        # the panel, weights_source and the log show the NAME; a copied file
        # with a stale name: would label a different tune (review 2026-10-04)
        raise ValueError(f"{path.name}: name: {name!r} is not the file name "
                         f"{path.stem!r} — rename one so they agree")
    return MpcTuneProfile(
        name=name, path=str(path),
        sha1=hashlib.sha1(data).hexdigest(), apply=apply, compare=compare,
        record=dict(raw.get("record") or {}),
        created=str(raw.get("created") or ""),
        description=str(raw.get("description") or "").strip(),
        source=dict(raw.get("source") or {}))


# ------------------------------------------------------------------- compare
def _close(a, b, rtol=1e-6, atol=1e-9) -> bool:
    try:
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
    except (TypeError, ValueError):
        return False
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=rtol, atol=atol))


def station_compare(P, cfg) -> dict:
    """The ``compare:`` values of the RUNNING station: ``P`` is the dobmpc
    params module AFTER the plant overrides (what the solver was generated
    from), ``cfg`` the MpcConfig."""
    return {"rov_model": str(P.MODEL),
            "ctrl_hz": float(getattr(cfg, "ctrl_hz", 1.0 / float(P.DT_CTRL))),
            "N": int(P.MPC_N), "dt_s": float(P.DT_CTRL),
            "u_max": [float(v) for v in P.U_MAX],
            # params.py stores damping NEGATED; the file reads positive
            "linear_damping": [float(-v) for v in np.asarray(P.DL, float)],
            "quadratic_damping": [float(-v) for v in np.asarray(P.DNL, float)],
            "axis_gain": {k: float(v) for k, v in
                          dict(getattr(cfg, "axis_gain", {}) or {}).items()}}


def profile_warnings(profile: MpcTuneProfile, *, P, cfg) -> list[str]:
    """Every way the station cannot fly this profile EXACTLY as it was made:
    a ``compare:`` value that differs from the running solver / vehicle, or
    an ``apply:`` value the station clamps. Empty = it flies as saved."""
    have = station_compare(P, cfg)
    out = [f"compare.{k} missing — not checked" for k in COMPARE_KEYS
           if k not in profile.compare]
    for k, want in profile.compare.items():
        got = have.get(k)
        if k == "axis_gain":
            want = dict(want or {})
            diff = sorted(kk for kk in set(want) | set(got or {})
                          if not _close(want.get(kk, math.nan),
                                        (got or {}).get(kk, math.nan)))
            if diff:
                out.append("axis_gain " + ", ".join(
                    f"{kk} {want.get(kk)}→{(got or {}).get(kk)}" for kk in diff))
        elif k == "rov_model":
            if str(want) != str(got):
                out.append(f"rov_model {want}→{got}")
        elif not _close(want, got):
            out.append(f"{k} {_short(want)}→{_short(got)}")
    cap_station = float(getattr(cfg, "axis_cap", 1.0))
    if profile.apply["axis_cap"] > cap_station + 1e-12:
        # HwDobMpc.axis_cap goes through MpcWorker._axis_cap, where a
        # controller may only TIGHTEN the station's ceiling
        out.append(f"axis_cap {profile.apply['axis_cap']:g} > station "
                   f"{cap_station:g}: flies {cap_station:g}")
    slew_station = float(getattr(cfg, "axis_slew_per_s", 0.0) or 0.0)
    if slew_station > 0.0 and profile.apply["axis_slew_per_s"] > slew_station + 1e-12:
        # HwDobMpc.axis_slew_per_s is tighten-only too (effective_slew)
        out.append(f"axis_slew_per_s {profile.apply['axis_slew_per_s']:g} > "
                   f"station {slew_station:g}: flies {slew_station:g}")
    trim_cfg = float(getattr(cfg, "vehicle_net_buoyancy_n", 0.0))
    trim = float(profile.apply["vehicle_net_buoyancy_n"])
    if abs(trim - trim_cfg) > 1e-9:
        # The payload (peg or not) is a VEHICLE state: PID / mpc fly the
        # hw_mpc.yaml value, mpc_tuned this one. A difference is exactly the
        # "changed the yaml for the peg, mpc_tuned still flies the old trim"
        # trap (KNOWN_ISSUES 2026-10-04), so it is never a quiet OK.
        out.append(f"vehicle_net_buoyancy_n {trim:+g} N (profile, flown by "
                   f"mpc_tuned) ≠ hw_mpc.yaml {trim_cfg:+g} N (PID / mpc) — "
                   f"is the payload the same?")
    return out


def effective_slew(profile_slew: float, station_slew: float) -> float:
    """The slew a profile flies: tighten-only against the station's (0 =
    the station has none, so the profile's is the only limit)."""
    p, st = float(profile_slew), float(station_slew or 0.0)
    return p if st <= 0.0 else min(p, st)


def _short(v) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(f"{float(x):g}" for x in v) + "]"
    return f"{v}"


# ------------------------------------------------------------------ snapshot
def current_apply(cfg, P) -> dict:
    """The ``apply:`` values the station flies under mpc_tuned WITHOUT a
    profile — i.e. hw_mpc.yaml + params.py as loaded. What ``save`` writes."""
    from .path_cost import resolve_tune

    pol = dict(getattr(cfg, "policy", {}) or {})
    along = pol.get("along_scale")
    return {
        "Q": [float(v) for v in P.MPC_Q],
        "QN": [float(v) for v in P.MPC_QN],
        "R": [float(v) for v in P.MPC_R],
        "path_cost": resolve_tune(getattr(cfg, "mpc_tuned", None)),
        "policy": {"q_scale": float(pol.get("q_scale", 1.0)),
                   "along_scale": (None if along is None else float(along))},
        "vehicle_net_buoyancy_n": float(getattr(cfg, "vehicle_net_buoyancy_n", 0.0)),
        "axis_cap": float(cfg.axis_cap),
        "axis_slew_per_s": float(cfg.axis_slew_per_s),
    }


def _sha1_file(p: Path) -> str | None:
    try:
        return hashlib.sha1(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def snapshot(cfg, P, *, name: str, description: str = "",
             config_path=None, base: MpcTuneProfile | None = None) -> dict:
    """A profile dict (``write_profile`` puts it on disk). ``apply`` is
    ``base.apply`` when a base profile is given — what mpc_tuned flies while
    that profile is held — else hw_mpc.yaml + params.py (``current_apply``,
    what it flies with no profile). ``compare`` / ``record`` are always THIS
    station. ``P`` must already carry the plant overrides
    (``apply_plant_overrides``)."""
    raw = dict(getattr(cfg, "raw", {}) or {})
    pol_raw = dict(raw.get("policy") or {})
    cfg_p = Path(config_path) if config_path else None
    params_p = Path(getattr(P, "__file__", "") or "")
    rel = (lambda p: str(p.resolve().relative_to(REPO_ROOT))
           if p and p.resolve().is_relative_to(REPO_ROOT) else str(p))
    return {
        "schema": SCHEMA,
        "name": str(name),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "description": str(description),
        "source": {
            "config": (rel(cfg_p) if cfg_p else None),
            "config_sha1": (_sha1_file(cfg_p) if cfg_p else None),
            "params": (rel(params_p) if params_p.name else None),
            "params_sha1": (_sha1_file(params_p) if params_p.name else None),
            "git_commit": _git("rev-parse", "HEAD") or None,
            "git_dirty": bool(_git("status", "--porcelain", "--",
                                   "config/hw_mpc.yaml", "rov_gui/control",
                                   "bluerov2_mujoco_marinegym/dobmpc")),
            # where `apply` came from: a held profile (name + sha1), or the
            # config's own blocks
            "apply_from": ({"profile": base.name, "sha1": base.sha1}
                           if base is not None else "config"),
        },
        "apply": (_plain(base.apply) if base is not None
                  else current_apply(cfg, P)),
        "compare": station_compare(P, cfg),
        "record": {
            "note": "context only — never read back by the station",
            "model": {
                "mass_kg": float(P.MASS),
                "inertia": [float(P.IX), float(P.IY), float(P.IZ)],
                "added_mass": [float(v) for v in P.ADDED_MASS],
                "volume_m3": float(P.VOLUME), "rho": float(P.RHO),
                "zg_m": float(P.ZG),
                "net_buoyancy_model_n": float(P.NET_BUOYANCY),
                "v_max_m_s": float(P.V_MAX),
                "horizon_s": round(float(P.MPC_N) * float(P.DT_CTRL), 4),
            },
            "path": {k: raw.get(k) for k in (
                "path_following", "path_lead_m", "path_fillet_m",
                "path_turn_radius_m", "path_lat_accel_m_s2",
                "path_long_accel_m_s2", "path_creep_m_s")},
            # the hw_mpc.yaml blocks as written (null = not in the file)
            "hw_mpc_yaml": {
                "mpc_tuned": raw.get("mpc_tuned"),
                "plant": raw.get("plant"),
                "axis_gain": raw.get("axis_gain"),
                "axis_cap": raw.get("axis_cap"),
                "axis_slew_per_s": raw.get("axis_slew_per_s"),
                "vehicle_net_buoyancy_n": raw.get("vehicle_net_buoyancy_n"),
                "w_hat_clip": raw.get("w_hat_clip"),
                "policy.q_scale": pol_raw.get("q_scale"),
                "policy.along_scale": pol_raw.get("along_scale"),
                "policy.heave_trim_attitude_rotated":
                    pol_raw.get("heave_trim_attitude_rotated"),
            },
        },
    }


_HEADER = """\
# MPC_Tuned profile — rov_gui/control/mpc_tune.py (schema {schema}).
# Picked on the trajectory panel (LOW = MPC_Tuned, `…` on the tune row), or
# at launch: `--mpc-tune NAME` / hw_mpc.yaml `mpc_tuned_profile: NAME`.
#   apply:   APPLIED while LOW is mpc_tuned / mpc_tuned_7d (runtime, no rebuild).
#            Every key required. Edit these to make a new tune — and save it
#            under a NEW name: run metas pin the file's sha1.
#   compare: NOT applied; the solver/vehicle this tune was made on. A
#            difference from the running station is a warning (panel + meta).
#   record:  context only, never read back.
"""


def write_profile(d: dict, path, *, overwrite: bool = False) -> Path:
    """Write ``d`` (a ``snapshot``) to ``path`` and prove it loads back."""
    import yaml

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pick another name, or "
                              f"--overwrite)")
    resolve_apply(d["apply"])                       # never write a bad one
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(_plain(d), sort_keys=False, allow_unicode=True,
                          default_flow_style=None, width=100)
    path.write_text(_HEADER.format(schema=SCHEMA) + body, encoding="utf-8")
    load_profile(path)
    return path


def _plain(v):
    """numpy / tuples -> plain YAML/JSON types."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v
