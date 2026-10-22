#!/usr/bin/env python3
"""Integral-error observation on the RL pwm LOWs (2026-10-18, RL_controller pwm12 contract C6 — the "PI" variant).

An RL_controller export whose obs_spec says ``obs_integral.enabled: true`` was trained with 5 more frame floats between
the goal feed-forward and prev_action: a leaky, clamped integral of the position / roll-pitch error the frame already
shows (rov_rl/obs_integral.py ErrorIntegrator in the env; its numpy twin deploy/obs_builder.py ErrorIntegratorNp; the
round-2 options att_error "euler_rp" and cond_pos_m / cond_att_rad exist only in the dev copy
logs/r2dev_20261018/src). rl_policy.py builds it in HwRl._frame (StationErrorIntegrator) for rl_pwm / rl_pwm_7d only.

Pinned here:
  (a) parity: HwRlPwm / HwRlPwm7d.observation == RL_controller deploy ObsBuilder (<= 1e-12) over a random sequence of
      states, goals and attitude references that never trips the leash, across an engage, a re-engage and a third
      engagement (the first drives I_pos and I_att into their clamps, the second straddles the round-2 conditional
      thresholds, the third has headings anywhere) — for the round-1 law (live tree; also against the dev copy, whose
      defaults are the round-1 law), euler_rp, quat_xy + cond, euler_rp + cond (dev copy), and a world-frame twin;
      against fixtures/rl_integral_parity_20261018.json always, and against the builders themselves (exec'd from
      source, read-only) when the RL_controller checkout is on this machine;
  (b) the first frame after a reset carries exactly ONE update from zero (masked by cond), tiled over the history;
  (c) saturation passes through unclipped: int_att_rp at exactly +-1, int_pos_b at sqrt(2) (level) / sqrt(3) (tilted)
      in the network's input, with finite pulses;
  (d) every refusal: a non-dict block, a non-boolean / missing enabled, missing / non-finite / non-positive tau_s,
      pos_max, att_max, tau_s <= control_dt, an unknown att_error, negative / non-finite / non-numeric cond_*, slots
      not exactly INTEGRAL_SLOTS, an unknown key (enabled or not), an integral layout without enabled true, enabled
      true with a plain layout, a frame_dim that is not the layout's width, obs_offset_clip <= 0, and an integral export
      on rl / rl_7d — ValueError at build, naming the LOW, the export and the reason;
  (e) every export without the block (or with enabled false) is BIT-identical to the module before this change:
      obs fed to the network, u, pulses, prev_action, last_action, info, meta and the build log, over random runs
      with a re-engage, the tag gate's HOLD, a stale fix, coast ticks and far goals — against the frozen pre-change
      copy (RL_controller logs/r2dev_20261018/station_port/rl_policy_pre_20261018.py) when it is here, against
      fixtures/rl_integral_legacy_golden_20261018.json (recorded FROM that frozen copy; it stores the ticks it was
      recorded on and is compared bit for bit in the recording's numerics environment, floats to 1e-12 elsewhere —
      follow-up, review T6) always, plus the existing goldens (fixtures/rl_pwm_trim0_golden_20261018.json,
      fixtures/rl_goalframe_legacy_obs.npz);
  (f) the real pretest PI export (iteration 200) builds on BOTH pwm LOWs with the shipped config/hw_mpc.yaml (only the
      policy_dir pointed at a tmp copy) and flies 50 ticks with finite pulses (skipped when the export is absent);
  (g) fix round 2026-10-18 (review F1, the override rule): the update that follows a command which was not the
      policy's own masks its add and keeps the leak — after the tag gate's HOLD pulses (tick for tick: equal to a run
      without the HOLD through the first HOLD tick, then exactly the leak until the first update after the HOLD, then
      the same adds again), after nothing was sent (note_applied_pwm(None): the worker's coast and pwm-gate grace), after
      a failed tick; the finding's 3 s thin-fix HOLD 0.15 m below the goal no longer winds the integral up;
  (h) (review F2) step's info carries, per tick, the slots the network saw and the mask (obs_int_slots,
      obs_int_add_masked; integral exports only), and a re-engage keeps the finished engagement's record in meta;
  (i) (review F4) the update_law / units texts are RL_controller's (fixtures/rl_integral_law_texts_20261018.json
      always, the RL generators when they are here) and any other text is refused at build; (review F3) the build line
      says which round-2 keys were defaulted;
  (j) follow-up 2026-10-18 (the round-1 review findings the fix round did not get): the integral takes the GUARDED
      position — a thin-fix jump PwmObsGuard rejects never reaches it (T1); e_pos is clipped to obs_offset_clip where
      the leash leaves a component above it (leash_m 2.0, or obs_offset_clip 0.5), and the cond mask judges the
      clipped e (T2); the cond mask is STRICT at |e| == cond, per component, at integrator level (against the dev
      ErrorIntegratorNp) and through the follower (T3 / parity F1); euler_rp wraps the roll difference across +-180 deg
      (T4 / parity F2); what the network receives INSIDE step() is what observation() builds on the flown reference and
      the fed-back action (any machine) and what RL_controller's builders build (where they are here) (parity F3);
  plus: the leash convention (e_pos = the LEASHED goal offset; beyond the leash a minor axis integrates LESS than
  training would, review F5), a non-finite tick neither raises nor poisons the integral, the meta block / build line,
  and frame_layout == RL_controller rov_rl/obs_spec.frame_layout.

Fixture provenance: RL_controller logs/r2dev_20261018/station_port/make_integral_fixtures.py (it imports the drivers
below, so the fixtures and these tests make the same calls; each fixture carries a ``meta`` block with the sha256 of
what produced it). Re-generate only when the RL builders' law changes (parity) or never (legacy golden: it is the
pre-change module's output; re-recording it from the new module would defeat it). Fix round 2026-10-18: the parity
fixture's obs_integral blocks were re-stamped with the round-2 update_law texts (frames verified unchanged,
station_port/restamp_parity_blocks_20261018.py); fixtures/rl_integral_law_texts_20261018.json is
station_port/make_law_texts_fixture_20261018.py's record of the RL text generators. Follow-up 2026-10-18 (review T6):
the legacy golden stores each run's input ticks and meta.recorded_env (numerics_env) — station_port/followup/
make_legacy_golden_inputs_20261018.py re-recorded every run from the FROZEN module on the stored ticks and wrote only
after all of its outputs were byte-identical to the file it replaced.

Run (both station envs; env robust has numpy 1.26.4, rovgui-pose 2.4.4):
  QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python -m pytest rov_gui/tests/test_rl_integral_obs.py -q
  QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest rov_gui/tests/test_rl_integral_obs.py -q
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rov_gui.control.geometry import MpcConfig  # noqa: E402
from rov_gui.control.path_geometry import NedPlan  # noqa: E402
from rov_gui.control.rl_policy import (INTEGRAL_SLOTS, OBS_INTEGRAL_ENGAGEMENTS_KEPT,  # noqa: E402
                                       OBS_INTEGRAL_UNITS_TEXT, HwRl, HwRl7d, HwRlPwm, HwRlPwm7d,
                                       StationErrorIntegrator, check_frame_layout, check_obs_integral, frame_layout,
                                       obs_integral_law_text)
from rov_gui.control.rl_pwm import GUARD_HOLD, GUARD_JUMP, GUARD_NONE, NEUTRAL_PULSES  # noqa: E402

FIX = HERE / "fixtures"
POLICIES = ROOT / "rov_gui" / "control" / "rl_policies"
#: verbatim copy of RL_controller logs/r2dev_20261018/pretest/runs/2026-10-18_00-00-01_pretestPI/exported/obs_spec.json
PI_SPEC = FIX / "rl_integral_pretestPI_obs_spec_20261018.json"
PARITY = FIX / "rl_integral_parity_20261018.json"
LEGACY = FIX / "rl_integral_legacy_golden_20261018.json"
#: fix round 2026-10-18 (review F4): RL_controller's update_law / units texts, recorded from the RL sources
LAW_TEXTS = FIX / "rl_integral_law_texts_20261018.json"
#: read-only, and only when present on this machine
RLC = Path("/home/bdml/Desktop/RL_controller")
R2DEV = RLC / "logs" / "r2dev_20261018"
RLC_LIVE_BUILDER = RLC / "deploy" / "obs_builder.py"
RLC_DEV_BUILDER = R2DEV / "src" / "deploy" / "obs_builder.py"
RLC_LIVE_OBS_SPEC = RLC / "rov_rl" / "obs_spec.py"
RLC_DEV_OBS_SPEC = R2DEV / "src" / "rov_rl" / "obs_spec.py"
RLC_LIVE_EXPORTER = RLC / "scripts" / "export_policy.py"
RLC_DEV_EXPORTER = R2DEV / "src" / "scripts" / "export_policy.py"
PI_EXPORT = R2DEV / "pretest" / "runs" / "2026-10-18_00-00-01_pretestPI" / "exported"
#: rl_policy.py as it was before this port (copied 2026-10-18 before any edit; mtime 2026-10-18 03:19:49)
FROZEN = R2DEV / "station_port" / "rl_policy_pre_20261018.py"
FROZEN_SHA256 = "c761c26557e02e9ad68cc077127ecff2b9099458b0464bdd1e882bcb2d4c1b95"

DT = 0.05
LEAK = 1.0 - DT / 4.0                   # the pwm12 leak (tau 4 s)
#: station observation vs RL_controller's builders. Follow-up 2026-10-18 (review T5): 1e-9 -> 1e-12. Measured worst over
#: the 9 parity runs: 2.220e-16 against the fixture under numpy 1.26.4 (euler_rp), exactly 0 under 2.4.4 and against the
#: builders exec'd in either env [측정 RL_controller/logs/r2dev_20261018/station_port/followup/t5_parity_tol_robust.txt,
#: t5_parity_tol_rovgui-pose.txt]. Deliberately not 0: e_att from the world quaternions, an equivalent mutant (review T5),
#: moves it by up to 3.331e-16 [측정 RL_controller/logs/r2dev_20261018/station_port/verify_tests/t5_m40_err.txt].
TOL = 1e-12
LOWS_PWM = ("rl_pwm", "rl_pwm_7d")
CLS = {"rl": HwRl, "rl_7d": HwRl7d, "rl_pwm": HwRlPwm, "rl_pwm_7d": HwRlPwm7d}
#: the slots in a pwm frame with the goal feed-forward: q_des 0:4, offset_b 4:7, q_est 7:11, v_b 11:14, gyro 14:17,
#: goal_v_b 17:20, goal_yaw_rate 20, int_pos_b 21:24, int_att_rp 24:26, prev_action 26:34
SL_POS, SL_ATT, SL_INT = slice(21, 24), slice(24, 26), slice(21, 26)
FRAME_HEAD = 26
_DROP = object()


# =============================================================================
# helpers
# =============================================================================
def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_source(path: Path, name: str, package: str | None = None):
    """``path`` executed as module ``name`` WITHOUT the import system (no bytecode is written next to it: the
    RL_controller tree is read-only here). ``package`` = the package its relative imports resolve in."""
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    if package:
        mod.__package__ = package
        sys.modules[name] = mod
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    return mod


@functools.lru_cache(maxsize=None)
def _rlc_builder(which: str):
    """RL_controller deploy/obs_builder.py — "live" (round-1 law) or "dev" (r2dev copy: + euler_rp, cond) — or None."""
    p = RLC_LIVE_BUILDER if which == "live" else RLC_DEV_BUILDER
    return _load_source(p, f"rlc_obs_builder_{which}") if p.is_file() else None


@functools.lru_cache(maxsize=None)
def _frozen():
    """The pre-change rl_policy.py, run INSIDE rov_gui.control so its relative imports (.pid, .rl_pwm, .attitude_ref,
    .geometry) are today's modules — the comparison isolates rl_policy.py — with REPO_ROOT set to this checkout, as if
    it still lay in rov_gui/control/: policy dirs, meta and build lines then compare as strings. None when absent."""
    if not FROZEN.is_file():
        return None
    assert _sha256(FROZEN) == FROZEN_SHA256, "the frozen pre-change copy was modified: it no longer proves anything"
    mod = _load_source(FROZEN, "rov_gui.control._rl_policy_pre_integral_20261018", package="rov_gui.control")
    mod.REPO_ROOT = str(ROOT)
    return mod


def _pi_spec() -> dict:
    return json.loads(PI_SPEC.read_text(encoding="utf-8"))


def _spec(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _copy(d: dict) -> dict:
    return json.loads(json.dumps(d))


def _world_twin(spec: dict) -> dict:
    """The goal-frame export's world-frame twin, spelled out here (not via rl_policy.frame_layout)."""
    s = _copy(spec)
    ren = {"q_des_goalframe(w,x,y,z)": "q_des(w,x,y,z)", "q_est_goalframe(w,x,y,z)": "q_est(w,x,y,z)"}
    s["frame_layout"] = [ren.get(x, x) for x in s["frame_layout"]]
    s["obs_goal_frame"] = False
    return s


@functools.lru_cache(maxsize=None)
def _law_texts() -> dict:
    return json.loads(LAW_TEXTS.read_text(encoding="utf-8"))


def law_text_rlc(tau_s=4.0, pos_max=0.4, att_max=0.8, att_error="quat_xy", cond_pos_m=0.0, cond_att_rad=0.0) -> str:
    """The update_law RL_controller's generator writes for this law — from fixtures/rl_integral_law_texts_20261018.json,
    recorded from the RL sources (never from the station's own transcription)."""
    key = (float(tau_s), float(pos_max), float(att_max), str(att_error), float(cond_pos_m), float(cond_att_rad))
    for r in _law_texts()["texts"]:
        if (r["tau_s"], r["pos_max"], r["att_max"], r["att_error"], r["cond_pos_m"], r["cond_att_rad"]) == key:
            return r["dev"]
    raise KeyError(f"no recorded RL_controller text for {key}: add it to make_law_texts_fixture_20261018.py GRID")


def with_rlc_text(block: dict) -> dict:
    """``block`` with the update_law RL_controller writes for the law it declares (absent round-2 keys = round 1) — what
    an export of that law carries, so the station's text gate accepts it (review F4)."""
    b = dict(block)
    b["update_law"] = law_text_rlc(b["tau_s"], b["pos_max"], b["att_max"], b.get("att_error", "quat_xy"),
                                   b.get("cond_pos_m", 0.0), b.get("cond_att_rad", 0.0))
    return b


def _exporter_units(path: Path) -> str:
    """The literal "units" value of RL_controller scripts/export_policy._integral_block, read with ast (the exporter
    imports torch / yaml and is never executed here)."""
    import ast
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == "_integral_block":
            for d in ast.walk(node):
                if isinstance(d, ast.Dict):
                    for k, v in zip(d.keys, d.values):
                        if isinstance(k, ast.Constant) and k.value == "units":
                            return ast.literal_eval(v)
    raise AssertionError(f"no _integral_block units literal in {path}")


#: The parity cases: name -> (obs_integral keys over the pretest export's round-1 block, goal frame, the RL builder
#: whose numbers the fixture holds). The round-1 export carries no att_error / cond keys (the defaulted path).
CASES = {
    "r1_quat_xy": ({}, True, "live"),
    "r2_euler_rp": ({"att_error": "euler_rp", "cond_pos_m": 0.0, "cond_att_rad": 0.0}, True, "dev"),
    "r2_quat_xy_cond": ({"att_error": "quat_xy", "cond_pos_m": 0.1, "cond_att_rad": 0.3}, True, "dev"),
    "r2_euler_rp_cond": ({"att_error": "euler_rp", "cond_pos_m": 0.1, "cond_att_rad": 0.3}, True, "dev"),
    "r1_world_frame": ({}, False, "live"),
}
RUNS = [(c, lo) for c in CASES for lo in LOWS_PWM if not (c == "r1_world_frame" and lo == "rl_pwm")]
PARITY_SEEDS = {"rl_pwm": 20261018, "rl_pwm_7d": 20261019}
N_PARITY = 52
PARITY_RESETS = (0, 26, 42)


def case_spec(case: str) -> dict:
    """The pretest export with the case's law keys — and (fix round, review F4) the update_law text RL_controller writes
    for that law (the dev generator's round-2 clauses), as a real export of it carries."""
    over, gf, _ = CASES[case]
    s = _pi_spec()
    s["obs_integral"] = with_rlc_text({**s["obs_integral"], **over})
    return s if gf else _world_twin(s)


def _export(tmp, spec: dict, name: str = "export", weights_from: Path | None = None) -> Path:
    """An export dir with ``spec``: the weights of ``weights_from`` (a real export of the same shape), else a FORMAT
    zero-weight MLP (never a policy; the observation does not read the weights; constant actions)."""
    pdir = Path(tmp) / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "obs_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    if weights_from is not None:
        shutil.copy(Path(weights_from) / "policy_weights.npz", pdir / "policy_weights.npz")
        return pdir
    n_in, n_act = int(spec["frame_dim"]) * int(spec["history_len"]), int(spec["num_actions"])
    np.savez(pdir / "policy_weights.npz", n_layers=np.array(2), activation=np.array("elu"),
             W0=np.zeros((8, n_in), np.float32), b0=np.zeros(8, np.float32),
             W1=np.zeros((n_act, 8), np.float32), b1=np.linspace(-0.3, 0.3, n_act).astype(np.float32))
    return pdir


def _cfg(low: str, pdir, **block) -> MpcConfig:
    """MpcConfig flying ``pdir`` on ``low``; the pwm transport block at cap 1.0 (min(cap, trained) flies) with the
    sign-probe gate opted out (offline; test_rl_pwm_scale.py owns it)."""
    cfg = MpcConfig()
    cfg.rl_pwm = {"pwm_cap": 1.0, "require_thruster_probe": False}
    if low == "rl_pwm":
        cfg.rl_pwm.update({"policy_dir": str(pdir), **block})
    else:
        setattr(cfg, low, {"policy_dir": str(pdir), **block})
    return cfg


def _build(low: str, pdir, mod=None, log=None, **block):
    cls = getattr(mod, CLS[low].__name__) if mod is not None else CLS[low]
    return cls(_cfg(low, pdir, **block), log=log if log is not None else (lambda m: None))


def _R(q) -> np.ndarray:
    """Rotation matrix (body -> world) of a unit (w,x,y,z) quaternion — written out here, independent of the module."""
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _R_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) Ry(pitch) Rx(roll), z-up / FLU — the body->world rotation q_est stands for, from first principles."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _hamilton(a, b) -> np.ndarray:
    """a (x) b for (w,x,y,z), via the vector form (independent of the module's component formula)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = a[0] * b[0] - float(np.dot(a[1:], b[1:]))
    v = a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])
    return np.r_[w, v]


def _canon(x):
    """JSON-comparable form (numpy -> python, tuples -> lists) — compared via json.dumps(sort_keys=True)."""
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, np.ndarray):
        return _canon(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _dump(x) -> str:
    return json.dumps(_canon(x), sort_keys=True)


def _norm_log(lines) -> list:
    """Build lines with the one timing number (the MLP probe) masked."""
    return [re.sub(r"probe \d+\.\d+ ms", "probe <t> ms", str(m)) for m in lines]


def history_from_frames(frames: np.ndarray, resets, K: int) -> np.ndarray:
    """The newest-first history every builder here assembles from its frames: refilled with the frame of an engage
    tick, rolled one frame per tick after it."""
    rs = {int(r) for r in resets}
    out, r = [], 0
    for k in range(len(frames)):
        if k in rs:
            r = k
        out.append(np.concatenate([frames[max(k - j, r)] for j in range(K)]))
    return np.array(out)


# =============================================================================
# (a) parity drivers — shared with the fixture generator
# =============================================================================
def parity_inputs(seed: int, tilted: bool, n: int = N_PARITY, resets=PARITY_RESETS) -> dict:
    """Deterministic parity ticks (stored in the fixture, so numpy's random stream never matters there): engagement
    A [resets[0], resets[1]) holds a persistent goal offset (0.48, -0.44, 0.38) m z-up and a large attitude error
    (NED roll -0.70, pitch +0.80 against a goal of roll 0.10 / pitch -0.20 or level), so I_pos and I_att reach their
    clamps; B [resets[1], resets[2]) has errors straddling the round-2 thresholds 0.1 m / 0.3 rad; C has headings
    anywhere (heading errors up to 180 deg). |goal - p| <= 0.75 m < leash_m 0.8 always (the leash is the one station
    convention the builders do not have). The reference yaw rate is never 0 and below yaw_rate_max (no recovery, no
    clip); ``rp`` is the flown roll/pitch reference (zeros for the level LOW)."""
    rng = np.random.default_rng(int(seed))
    r0, r1, r2 = (int(r) for r in resets)
    t = DT * np.arange(n)
    eta = np.zeros((n, 6))
    d_up = np.zeros((n, 3))
    yaw_ref = np.zeros(n)
    rp = np.zeros((n, 2))
    for k in range(n):
        if k < r1:
            eta[k] = np.array([0.3, -0.2, 0.9, -0.70, 0.80, 1.0]) + rng.normal(0.0, 1.0, 6) * np.r_[[0.004] * 3, [0.01] * 3]
            d_up[k] = np.array([0.48, -0.44, 0.38]) + rng.normal(0.0, 0.004, 3)
            yaw_ref[k] = 1.25 + rng.normal(0.0, 0.01)
            rp[k] = (np.array([0.10, -0.20]) + rng.normal(0.0, 0.005, 2)) if tilted else 0.0
        elif k < r2:
            eta[k] = np.r_[rng.uniform(-0.5, 0.5, 2), rng.uniform(0.5, 1.2), rng.uniform(-0.35, 0.35, 2),
                           rng.uniform(-0.6, 0.6)]
            d_up[k] = rng.normal(0.0, 0.12, 3)
            yaw_ref[k] = eta[k, 5] + rng.normal(0.0, 0.2)
            rp[k] = np.r_[rng.uniform(-0.12, 0.12), rng.uniform(-0.24, 0.24)] if tilted else 0.0
        else:
            eta[k] = np.r_[rng.uniform(-0.5, 0.5, 2), rng.uniform(0.5, 1.2), rng.uniform(-0.45, 0.45, 2),
                           rng.uniform(-math.pi, math.pi)]
            u = rng.normal(size=3)
            d_up[k] = u / np.linalg.norm(u) * rng.uniform(0.0, 0.75)
            yaw_ref[k] = rng.uniform(-math.pi, math.pi)
            rp[k] = np.r_[rng.uniform(-0.12, 0.12), rng.uniform(-0.24, 0.24)] if tilted else 0.0
        nd = float(np.linalg.norm(d_up[k]))
        if nd > 0.75:
            d_up[k] *= 0.75 / nd
    p_ref = eta[:, :3] + d_up * np.array([1.0, -1.0, -1.0])        # z-up -> NED: the mirror is its own inverse
    return {"t": t, "eta": eta, "rates": rng.uniform(-0.4, 0.4, (n, 3)), "v_w": rng.uniform(-0.3, 0.3, (n, 3)),
            "p_ref": p_ref, "yaw_ref": yaw_ref, "v_ref": rng.uniform(-0.15, 0.15, (n, 3)),
            "r_ref": rng.uniform(0.05, 0.5, n) * rng.choice([-1.0, 1.0], n), "rp": rp,
            "prev": rng.uniform(-1.0, 1.0, (n, 8)), "resets": [int(r) for r in resets]}


def builder_run(ob, spec: dict, inp: dict) -> dict:
    """RL_controller's ObsBuilder over ``inp`` -> {"obs" (n, obs_dim) float64 = its own newest-first history,
    "frames" (n, frame_dim), "nu" (n, 6)}. Per tick: reset at the engage ticks, the goal through set_goal (NED, the
    tick's roll/pitch reference), a fix captured at the tick's own time (age 0: its estimator propagates nothing) at the
    tick's pose with its LPF world velocity set to the tick's ``v_w``; ``nu`` = its body velocity back in FRD + the
    rates, i.e. exactly the floats the station's state hands the follower. ObsBuilder.step returns float32: the float64
    history (``hist``) is what is compared."""
    b = ob.ObsBuilder(spec)
    rs = {int(r) for r in inp["resets"]}
    obs, frames, nus = [], [], []
    for k in range(len(inp["t"])):
        if k in rs:
            b.reset()
        eta, rates, t = np.asarray(inp["eta"][k], float), np.asarray(inp["rates"][k], float), float(inp["t"][k])
        b.set_goal(np.asarray(inp["p_ref"][k], float), float(inp["yaw_ref"][k]),
                   v_goal_ned=np.asarray(inp["v_ref"][k], float), yaw_rate_ned=float(inp["r_ref"][k]),
                   roll_ned=float(inp["rp"][k][0]), pitch_ned=float(inp["rp"][k][1]))
        b.fix_p, b.fix_yaw, b.fix_t = ob.C_NED_TO_UP @ eta[:3], -float(eta[5]), t
        b.v_w = np.array(inp["v_w"][k], float)
        b.step(t, None, {"roll": float(eta[3]), "pitch": float(eta[4]), "p": float(rates[0]), "q": float(rates[1]),
                         "r": float(rates[2])}, prev_action=np.asarray(inp["prev"][k], float))
        v_b = b.last["v_b"]
        nus.append([v_b[0], -v_b[1], -v_b[2], rates[0], rates[1], rates[2]])
        obs.append(np.array(b.hist, float).reshape(-1))
        frames.append(np.array(b.hist[0], float))
    return {"obs": np.array(obs), "frames": np.array(frames), "nu": np.array(nus, float)}


def station_run(ctrl, inp: dict, nu) -> np.ndarray:
    """HwRlPwm / HwRlPwm7d.observation over ``inp`` with ``nu``: reset at the engage ticks, the reference through
    set_target_ned, and (7d) the flown roll/pitch reference and the fed-back action set as the builder's."""
    rs = {int(r) for r in inp["resets"]}
    nu = np.asarray(nu, float)
    out = []
    for k in range(len(inp["t"])):
        if k in rs:
            ctrl.reset()
        ctrl.set_target_ned(np.asarray(inp["p_ref"][k], float), float(inp["yaw_ref"][k]),
                            v_ned=np.asarray(inp["v_ref"][k], float), r_ned=float(inp["r_ref"][k]))
        if isinstance(ctrl, HwRlPwm7d) or hasattr(ctrl, "_rp_flown"):
            ctrl._rp_flown = (float(inp["rp"][k][0]), float(inp["rp"][k][1]))
        ctrl._prev_action = np.array(inp["prev"][k], float)
        out.append(np.array(ctrl.observation(np.asarray(inp["eta"][k], float), nu[k], float(inp["t"][k])), float))
    return np.array(out)


@functools.lru_cache(maxsize=None)
def _parity_fixture() -> dict:
    return json.loads(PARITY.read_text(encoding="utf-8"))


def _fixture_inputs(low: str) -> dict:
    return {k: (np.asarray(v, float) if k != "resets" else list(v))
            for k, v in _parity_fixture()["inputs"][low].items()}


def _fixture_obs(case: str, low: str, K: int) -> np.ndarray:
    """The builder's history for one fixture run: its frames (prev_action left out of the file: it is the input
    ``prev``, clipped to [-1, 1] by the builder) reassembled newest first."""
    run = _parity_fixture()["runs"][f"{case}__{low}"]
    inp = _parity_fixture()["inputs"][low]
    frames = np.concatenate([np.asarray(run["frames_head"], float), np.clip(np.asarray(inp["prev"], float), -1, 1)],
                            axis=1)
    return history_from_frames(frames, inp["resets"], K)


# =============================================================================
# (a) parity
# =============================================================================
def test_parity_fixture_is_this_export_and_these_inputs():
    """The fixture was made from the export copied in PI_SPEC and the inputs parity_inputs makes (the inputs are stored,
    so a different numpy random stream only shows here, never in the parity itself)."""
    fx = _parity_fixture()
    assert fx["meta"]["pi_spec_sha256"] == _sha256(PI_SPEC)
    assert sorted(fx["runs"]) == sorted(f"{c}__{lo}" for c, lo in RUNS)
    for low in LOWS_PWM:
        got = _fixture_inputs(low)
        assert got["resets"] == list(PARITY_RESETS) and len(got["t"]) == N_PARITY
        if np.__version__ == fx["meta"]["numpy"]:          # another numpy may draw another stream: not a defect
            mine = parity_inputs(PARITY_SEEDS[low], tilted=(low == "rl_pwm_7d"))
            for k in ("t", "eta", "rates", "v_w", "p_ref", "yaw_ref", "v_ref", "r_ref", "rp", "prev"):
                assert np.array_equal(mine[k], got[k]), (low, k)
        d_up = (got["p_ref"] - got["eta"][:, :3]) * np.array([1.0, -1.0, -1.0])
        assert np.linalg.norm(d_up, axis=1).max() < 0.8                         # never trips the station's leash
        assert np.all(np.abs(got["r_ref"]) > 0.0) and np.abs(got["r_ref"]).max() < 0.6
    for case, low in RUNS:
        assert fx["runs"][f"{case}__{low}"]["obs_integral"] == case_spec(case)["obs_integral"], (case, low)


@pytest.mark.parametrize("case,low", RUNS, ids=[f"{c}-{lo}" for c, lo in RUNS])
def test_station_observation_equals_the_rl_builder_fixture(case, low, tmp_path):
    inp = _fixture_inputs(low)
    ctrl = _build(low, _export(tmp_path, case_spec(case), "pi"))
    got = station_run(ctrl, inp, inp["nu"])
    want = _fixture_obs(case, low, ctrl.K)
    assert got.shape == want.shape == (N_PARITY, 136)
    err = float(np.abs(got - want).max())
    assert err <= TOL, (case, low, err)
    # what the run covers: without the cond mask the first engagement drives I into its clamps — int_att_rp at exactly
    # +-1, int_pos_b beyond 1 (unclipped) — and the slots move on every tick; with it, nothing is added there (every
    # |e| is far above 0.1 m / 0.3 rad) and the slots stay exactly 0 until the second engagement
    sl = got[:, SL_INT]
    a = slice(0, PARITY_RESETS[1])
    if CASES[case][0].get("cond_pos_m", 0.0) == 0.0:
        assert np.any(np.abs(sl[a, 3:5]) == 1.0), case
        assert np.abs(sl[a, 0:3]).max() > 1.0, case
        assert np.all(np.abs(np.diff(sl, axis=0)).max(axis=1) > 0.0), case
    else:
        assert np.all(sl[a, 0:3] == 0.0) and np.any(sl[PARITY_RESETS[1]:] != 0.0), case


@pytest.mark.parametrize("low", LOWS_PWM)
def test_the_conditional_mask_and_the_euler_law_change_the_slots(low):
    """The fixture runs differ where they must: the cond mask holds the add back (A: position errors far above 0.1 m
    -> int_pos_b stays 0), euler_rp differs from quat_xy in int_att_rp only, the world-frame twin (rl_pwm_7d) keeps
    int_pos_b and offset_b (world-frame either way) and differs in the quaternion-derived slots."""
    runs = _parity_fixture()["runs"]
    o = {c: _fixture_obs(c, low, 4)[:, :34] for c in CASES if f"{c}__{low}" in runs}
    a, rest = slice(0, PARITY_RESETS[1]), slice(PARITY_RESETS[1], None)
    for c, plain in (("r2_quat_xy_cond", "r1_quat_xy"), ("r2_euler_rp_cond", "r2_euler_rp")):
        assert np.all(o[c][a, SL_POS] == 0.0), c              # every |e_pos| >= 0.1 m in A: nothing added
        assert np.any(o[c][rest, SL_POS] != 0.0) and np.any(o[c][rest, SL_ATT] != 0.0), c
        assert np.abs(o[c][:, SL_INT] - o[plain][:, SL_INT]).max() > 1e-3, c      # the mask changed the integral
    assert np.array_equal(o["r2_euler_rp"][:, SL_POS], o["r1_quat_xy"][:, SL_POS])
    assert np.abs(o["r2_euler_rp"][:, SL_ATT] - o["r1_quat_xy"][:, SL_ATT]).max() > 1e-3
    if "r1_world_frame" in o:
        assert np.array_equal(o["r1_world_frame"][:, SL_POS], o["r1_quat_xy"][:, SL_POS])
        assert np.array_equal(o["r1_world_frame"][:, 4:7], o["r1_quat_xy"][:, 4:7])
        assert not np.array_equal(o["r1_world_frame"][:, 0:4], o["r1_quat_xy"][:, 0:4])


@pytest.mark.parametrize("case,low", RUNS, ids=[f"{c}-{lo}" for c, lo in RUNS])
def test_station_observation_equals_the_rl_builders_live(case, low, tmp_path):
    """Against RL_controller's builders themselves (exec'd from source, read-only): the round-1 cases against the live
    tree AND the dev copy (whose defaults are the round-1 law), the round-2 cases against the dev copy — on the
    fixture's inputs (the fixture must still be what the builder says) and on a fresh random sequence."""
    which = ["live", "dev"] if CASES[case][2] == "live" else ["dev"]
    builders = [(w, _rlc_builder(w)) for w in which if _rlc_builder(w) is not None]
    if not builders:
        pytest.skip("RL_controller deploy/obs_builder.py not on this machine (the fixture test still pins the parity)")
    spec = case_spec(case)
    fx_frames = np.asarray(_parity_fixture()["runs"][f"{case}__{low}"]["frames_head"], float)
    for w, ob in builders:
        for seed, inp in (("fixture", _fixture_inputs(low)),
                          ("fresh", parity_inputs(PARITY_SEEDS[low] + 1000, tilted=(low == "rl_pwm_7d")))):
            ref = builder_run(ob, spec, inp)
            ctrl = _build(low, _export(tmp_path, spec, f"pi_{w}_{seed}"))
            got = station_run(ctrl, inp, ref["nu"])
            err = float(np.abs(got - ref["obs"]).max())
            assert err <= TOL, (case, low, w, seed, err)
            if seed == "fixture":                # the fixture is still what this builder says
                assert float(np.abs(ref["nu"] - inp["nu"]).max()) <= 1e-12, (case, low, w)
                assert float(np.abs(ref["frames"][:, :FRAME_HEAD] - fx_frames).max()) <= 1e-12, (case, low, w)


def test_frame_layout_is_rl_controllers_with_and_without_the_integral():
    mods = [(n, p) for n, p in (("live", RLC_LIVE_OBS_SPEC), ("dev", RLC_DEV_OBS_SPEC)) if p.is_file()]
    if not mods:
        pytest.skip("RL_controller rov_rl/obs_spec.py not on this machine")
    for n, p in mods:
        rlc = _load_source(p, f"rlc_obs_spec_{n}")
        assert tuple(rlc.INTEGRAL_SLOTS) == INTEGRAL_SLOTS
        for gv in (False, True):
            for na in (4, 6, 8):
                for gf in (False, True):
                    for integ in (False, True):
                        want = rlc.frame_layout(gv, na, goal_frame=gf, integral=integ)
                        assert frame_layout(gv, na, goal_frame=gf, integral=integ) == want, (n, gv, na, gf, integ)
                        assert len(want) == len(frame_layout(gv, na, goal_frame=gf, integral=integ))
                        assert rlc.frame_dim(gv, na, integral=integ) == 17 + 4 * gv + 5 * integ + na
    assert _pi_spec()["frame_layout"] == frame_layout(True, 8, goal_frame=True, integral=True)
    assert frame_layout(True, 8) == frame_layout(True, 8, integral=False)     # the old call is the old layout


# =============================================================================
# (b) the first frame after a reset
# =============================================================================
def _one_update(ctrl, frame: np.ndarray, eta, p_ref, oi: dict) -> np.ndarray:
    """One update from zero, from first principles: e_pos = clip(goal - p, +-1) (z-up), e_att from the frame's own
    quaternion slots (independent Hamilton product), the cond mask, the clamps, R^T by a rotation matrix."""
    d = (np.asarray(p_ref, float) - np.asarray(eta[:3], float)) * np.array([1.0, -1.0, -1.0])
    assert np.linalg.norm(d) < ctrl.leash_m
    e_pos = np.clip(d, -1.0, 1.0)
    q_d, q_o = frame[0:4], frame[7:11]
    q_err = _hamilton(np.r_[q_o[0], -q_o[1:]], q_d)
    e_att = 2.0 * np.sign(q_err[0]) * q_err[1:3]
    add_p, add_a = DT * e_pos, DT * e_att
    if oi.get("cond_pos_m", 0.0) > 0.0:
        add_p = np.where(np.abs(e_pos) < oi["cond_pos_m"], add_p, 0.0)
    if oi.get("cond_att_rad", 0.0) > 0.0:
        add_a = np.where(np.abs(e_att) < oi["cond_att_rad"], add_a, 0.0)
    i_pos, i_att = np.clip(add_p, -0.4, 0.4), np.clip(add_a, -0.8, 0.8)
    R = _R_euler(float(eta[3]), -float(eta[4]), -float(eta[5]))
    return np.r_[R.T @ i_pos / 0.4, i_att / 0.8]


@pytest.mark.parametrize("low", LOWS_PWM)
@pytest.mark.parametrize("cond", [False, True], ids=["round1", "cond"])
def test_the_first_frame_after_a_reset_carries_exactly_one_update(low, cond, tmp_path):
    spec = _pi_spec()
    if cond:
        spec["obs_integral"] = with_rlc_text({**spec["obs_integral"], "att_error": "quat_xy", "cond_pos_m": 0.1,
                                              "cond_att_rad": 0.3})
    ctrl = _build(low, _export(tmp_path, spec, "pi"))
    oi = spec["obs_integral"]
    eta = np.array([0.2, -0.1, 0.8, 0.05, 0.45, 0.3])                 # pitch NED 0.45: an error above 0.3 rad
    p_ref = eta[:3] + np.array([0.05, -0.30, 0.08]) * np.array([1.0, -1.0, -1.0])   # z-up (0.05, -0.30, 0.08)
    nu = np.array([0.05, -0.02, 0.01, 0.02, -0.03, 0.04])
    for engagement in range(3):
        ctrl.reset()
        ctrl.set_target_ned(p_ref, 0.35, v_ned=np.array([0.03, -0.02, 0.0]), r_ned=0.1)
        obs = ctrl.observation(eta, nu, 0.0)
        fr = obs.reshape(ctrl.K, -1)
        assert all(np.array_equal(fr[0], fr[j]) for j in range(1, ctrl.K))      # the history is tiled with it
        want = _one_update(ctrl, fr[0], eta, p_ref, oi)
        assert np.abs(fr[0, SL_INT] - want).max() <= 1e-12, (engagement, fr[0, SL_INT], want)
        assert ctrl.meta()["obs_integral"]["counts"]["ticks"] == 1
        if cond:                                    # the masked components: y (0.30 m) and pitch (~0.45 rad) not added
            i_pos_w = _R_euler(float(eta[3]), -float(eta[4]), -float(eta[5])) @ (fr[0, SL_POS] * 0.4)
            assert abs(i_pos_w[1]) < 1e-15 and abs(i_pos_w[0]) > 1e-3 and abs(i_pos_w[2]) > 1e-3
            assert fr[0, 25] == 0.0 and fr[0, 24] != 0.0
        # the second tick: one more update (leak, add) from the first — NOT two from zero
        obs2 = ctrl.observation(eta, nu, DT).reshape(ctrl.K, -1)
        assert not np.array_equal(obs2[0, SL_INT], fr[0, SL_INT])
        assert np.array_equal(obs2[1], fr[0])
        # a far-off error over a few ticks before the next engagement: that integral must not survive the reset
        for k in range(5):
            ctrl.observation(eta + np.array([0.3, 0.3, -0.2, 0.2, -0.3, 0.5]), nu, (2 + k) * DT)


@pytest.mark.parametrize("low", LOWS_PWM)
def test_a_refilled_history_restarts_the_integral_even_without_reset(low, tmp_path):
    """The env's rule as HwRl._frame mirrors it: WHENEVER the history is refilled (here: dropped without reset(), as no
    station path does today), the integral restarts before that tick's update — the refill, not reset(), is the rule.
    Also: the world-frame twin's frame (no goal-frame rotation) carries the same one-update slots — the body-frame
    error rotation conj(q_o) (x) q_d does not depend on the common yaw pre-rotation [유도]."""
    spec = _pi_spec()
    ctrl = _build(low, _export(tmp_path, spec, "pi"))
    eta = np.array([0.1, 0.0, 0.8, -0.05, 0.2, -0.4])
    p_ref = eta[:3] + np.array([0.2, 0.1, -0.15])
    nu = np.zeros(6)
    ctrl.reset()
    ctrl.set_target_ned(p_ref, -0.1)
    for k in range(6):
        ctrl.observation(eta, nu, k * DT)
    assert np.abs(ctrl._integ.I_pos).min() > 0.0
    ctrl._hist = None                                           # a refill without reset()
    fr = ctrl.observation(eta, nu, 6 * DT).reshape(ctrl.K, -1)
    want = _one_update(ctrl, fr[0], eta, p_ref, spec["obs_integral"])
    assert np.abs(fr[0, SL_INT] - want).max() <= 1e-12
    w = _build(low, _export(tmp_path, _world_twin(spec), "piw"))
    w.reset()
    w.set_target_ned(p_ref, -0.1)
    fw = w.observation(eta, nu, 0.0).reshape(w.K, -1)
    assert np.abs(fw[0, SL_INT] - fr[0, SL_INT]).max() <= 1e-12
    assert np.abs(fw[0, 0:4] - fr[0, 0:4]).max() > 1e-3                 # while the quaternion slots differ


# =============================================================================
# (c) saturation passes through, unclipped
# =============================================================================
#: (name, eta NED, goal offset z-up, expected int_pos_b): body x along world (1,1,0)/sqrt2 (level, yaw_up 45 deg) or
#: (1,1,1)/sqrt3 (yaw_up 45 deg, pitch_up -asin(1/sqrt3) = nose UP 35.3 deg) with I_pos at its clamp on those axes
SAT_CASES = [
    ("level_sqrt2", np.array([0.1, 0.2, 0.8, 0.0, 0.0, -math.pi / 4]), np.array([0.5, 0.5, 0.0]),
     np.array([math.sqrt(2.0), 0.0, 0.0])),
    ("tilted_sqrt3", np.array([0.1, 0.2, 0.8, 0.0, math.asin(1.0 / math.sqrt(3.0)), -math.pi / 4]),
     np.array([0.45, 0.45, 0.45]), np.array([math.sqrt(3.0), 0.0, 0.0])),
]


@pytest.mark.parametrize("low", LOWS_PWM)
@pytest.mark.parametrize("name,eta,d_up,want", SAT_CASES, ids=[c[0] for c in SAT_CASES])
def test_saturated_slots_reach_the_network_unclipped(low, name, eta, d_up, want, tmp_path):
    ctrl = _build(low, _export(tmp_path, _pi_spec(), "pi", weights_from=PI_EXPORT if PI_EXPORT.is_dir() else None))
    seen = []
    act = ctrl.policy.act
    ctrl.policy.act = lambda o: (seen.append(np.array(o, float)), act(o))[1]
    ctrl.reset()
    ctrl.set_target_ned(eta[:3] + d_up * np.array([1.0, -1.0, -1.0]), float(eta[5]))
    for k in range(60):
        ctrl.note_fix(4, 0.05)
        u, info = ctrl.step(eta, np.zeros(6), np.zeros(6), k * DT)
        assert info["status"] == 0 and all(isinstance(p, int) for p in info["pwm_us"]), (k, info)
        ctrl.note_applied_pwm(info["pwm_us"])
    assert len(seen) == 60
    last = seen[-1][:34]
    assert np.abs(last[SL_POS] - want).max() <= 1e-9, (name, last[SL_POS])
    assert np.max(np.abs(last[SL_POS])) > 1.0                                    # beyond [-1, 1], as trained
    assert np.abs(ctrl._integ.I_pos[d_up > 0] - 0.4).max() == 0.0                # at the clamp, exactly
    if name.startswith("tilted"):
        assert np.max(np.abs(last[SL_ATT])) == 1.0                               # the 35 deg pitch error: I_att clamped
    c = ctrl.meta()["obs_integral"]["counts"]
    assert c["ticks"] == 60 and c["ticks_any_saturated"] > 30 and c["max_abs_slot"][0] == pytest.approx(want[0], abs=1e-9)


# =============================================================================
# (d) refusals
# =============================================================================
def _plain_pwm_spec() -> dict:
    """The pretest export without its integral: a goal-frame pwm export of before (29-float frame, no block)."""
    s = _pi_spec()
    s.pop("obs_integral")
    s["frame_layout"] = [x for x in s["frame_layout"] if x not in INTEGRAL_SLOTS]
    s["frame_dim"] = 29
    s["obs_dim"] = 116
    return s


def _oi(**over) -> dict:
    b = dict(_pi_spec()["obs_integral"])
    for k, v in over.items():
        if v is _DROP:
            b.pop(k, None)
        else:
            b[k] = v
    return b


#: (spec maker, the refusal's text) — on BOTH pwm LOWs
REFUSALS = [
    (lambda: {**_pi_spec(), "obs_integral": [1, 2]}, r"obs_integral block that is not \{enabled"),
    (lambda: {**_pi_spec(), "obs_integral": "on"}, r"obs_integral block that is not \{enabled"),
    (lambda: {**_pi_spec(), "obs_integral": None}, r"obs_integral block that is not \{enabled"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(enabled="true")}, r"obs_integral\.enabled 'true' — expected true or false"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(enabled=1)}, r"obs_integral\.enabled 1 — expected true or false"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(enabled=None)}, r"obs_integral\.enabled None — expected"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(enabled=_DROP)}, r"obs_integral\.enabled None — expected"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(tau_s=_DROP)}, r"obs_integral\.tau_s is missing"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(pos_max=_DROP)}, r"obs_integral\.pos_max is missing"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_max=_DROP)}, r"obs_integral\.att_max is missing"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(tau_s=float("nan"))}, r"obs_integral\.tau_s is nan, not a number"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(pos_max=float("inf"))}, r"obs_integral\.pos_max is inf, not a number"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_max="0.8")}, r"obs_integral\.att_max is '0\.8', not a number"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_max=True)}, r"obs_integral\.att_max is True, not a number"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(pos_max=0.0)}, r"obs_integral\.pos_max 0 is not > 0"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_max=-0.8)}, r"obs_integral\.att_max -0\.8 is not > 0"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(tau_s=-4.0)}, r"obs_integral\.tau_s -4 is not > 0"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(tau_s=0.05)}, r"tau_s 0\.05 s does not exceed the control step 0\.05 s"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(tau_s=0.01)}, r"tau_s 0\.01 s does not exceed the control step"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_error="euler_xyz")}, r"obs_integral\.att_error 'euler_xyz'"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_error="")}, r"obs_integral\.att_error ''"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_error=None)}, r"obs_integral\.att_error None"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_error=1)}, r"obs_integral\.att_error 1"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(cond_pos_m=-0.1)}, r"obs_integral\.cond_pos_m -0\.1 is negative"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(cond_att_rad=-1e-9)}, r"obs_integral\.cond_att_rad -1e-09 is negative"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(cond_pos_m=float("inf"))}, r"obs_integral\.cond_pos_m is inf"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(cond_att_rad=float("nan"))}, r"obs_integral\.cond_att_rad is nan"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(cond_pos_m=True)}, r"obs_integral\.cond_pos_m is True"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(cond_att_rad="0.3")}, r"obs_integral\.cond_att_rad is '0\.3'"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(slots=_DROP)}, r"obs_integral\.slots None != \["),
    (lambda: {**_pi_spec(), "obs_integral": _oi(slots=[])}, r"obs_integral\.slots \[\] != \["),
    (lambda: {**_pi_spec(), "obs_integral": _oi(slots=["int_att_rp(2)", "int_pos_b(3)"])}, r"obs_integral\.slots \["),
    (lambda: {**_pi_spec(), "obs_integral": _oi(slots=list(INTEGRAL_SLOTS) + ["int_yaw(1)"])}, r"obs_integral\.slots"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(integrate_yaw=True)},
     r"obs_integral carries key\(s\) \['integrate_yaw'\] this station does not implement"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_error="euler_rp", reset_on_hold=False)},
     r"carries key\(s\) \['reset_on_hold'\]"),
    (lambda: {**_plain_pwm_spec(), "obs_integral": {**_oi(enabled=False, slots=[]), "anti_windup": "x"}},
     r"carries key\(s\) \['anti_windup'\]"),                                   # unknown key refused enabled or not
    (lambda: {**_pi_spec(), "obs_integral": _oi(enabled=False, slots=[])},
     r"frame_layout carries the integral-error slots .* but obs_integral\.enabled is false"),
    (lambda: {k: v for k, v in _pi_spec().items() if k != "obs_integral"},
     r"frame_layout carries the integral-error slots .* but obs_integral\.enabled is absent"),
    (lambda: {**_plain_pwm_spec(), "obs_integral": _oi()},
     r"says obs_integral\.enabled: true but its frame_layout has no integral-error slots"),
    (lambda: {**_pi_spec(), "frame_dim": 29}, r"frame_dim 29 != 34"),
    (lambda: {**_plain_pwm_spec(), "frame_dim": 34}, r"frame_dim 34 != 29"),
    (lambda: {**_pi_spec(), "frame_layout": _pi_spec()["frame_layout"][:7] + ["int_pos_b(3)", "prev_action(8)"]},
     r"frame_layout \[.*\] != \[.*obs_integral enabled\); refused"),
    (lambda: {**_pi_spec(), "obs_offset_clip": 0.0}, r"obs_offset_clip 0 is not > 0"),
    # fix round 2026-10-18 (review F4): the update_law / units texts are the contract
    (lambda: {**_pi_spec(), "obs_integral": _oi(update_law=_DROP)},
     r"obs_integral\.update_law is not the text RL_controller writes .* it is missing; refused"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(update_law=7)}, r"obs_integral\.update_law is not the text .* it is int 7"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(update_law=_pi_spec()["obs_integral"]["update_law"].replace(
        "(leak, then add, then clip, per component)", "(clip, then leak, then add, per component)"))},
     r"obs_integral\.update_law is not the text .* first difference at character \d+"),          # another ORDER
    (lambda: {**_pi_spec(), "obs_integral": _oi(update_law=_pi_spec()["obs_integral"]["update_law"].replace(
        "int_pos_b(3) = R(q_est_w)^T I_pos", "int_pos_b(3) = R(q_des_f)^T I_pos"))},
     r"obs_integral\.update_law is not the text"),                                               # another FRAME
    (lambda: {**_pi_spec(), "obs_integral": _oi(update_law=_pi_spec()["obs_integral"]["update_law"] + " ")},
     r"obs_integral\.update_law is not the text"),                                     # a trailing space: refused too
    (lambda: {**_pi_spec(), "obs_integral": _oi(tau_s=2.0)},
     r"obs_integral\.update_law is not the text .*\(tau 2 s, "),                     # a number its text contradicts
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_error="euler_rp", cond_pos_m=0.0, cond_att_rad=0.0)},
     r"obs_integral\.update_law is not the text .*att_error euler_rp"),                  # round-2 law, round-1 text
    (lambda: {**_pi_spec(), "obs_integral": _oi(update_law=law_text_rlc(att_error="euler_rp"))},
     r"obs_integral\.update_law is not the text .*att_error quat_xy"),                   # round-2 text, no round-2 keys
    (lambda: {**_pi_spec(), "obs_integral": _oi(att_error="quat_xy", cond_pos_m=0.1, cond_att_rad=0.3,
                                                update_law=law_text_rlc(cond_pos_m=0.1, cond_att_rad=0.3).replace(
                                                    "attitude 0.3 rad", "attitude 0.2 rad"))},
     r"obs_integral\.update_law is not the text .*cond 0\.1 / 0\.3"),                   # a threshold its text contradicts
    (lambda: {**_pi_spec(), "obs_integral": _oi(units=_DROP)}, r"obs_integral\.units is not the text .* it is missing"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(units=_pi_spec()["obs_integral"]["units"].replace(
        "PER WORLD AXIS", "PER BODY AXIS"))}, r"obs_integral\.units is not the text"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(implementations=_DROP)}, r"obs_integral\.implementations None — expected"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(implementations="  ")}, r"obs_integral\.implementations '  ' — expected"),
    (lambda: {**_pi_spec(), "obs_integral": _oi(implementations=["rov_rl/obs_integral.py"])},
     r"obs_integral\.implementations \['rov_rl/obs_integral\.py'\] — expected"),
]


@pytest.mark.parametrize("low", LOWS_PWM)
@pytest.mark.parametrize("i", range(len(REFUSALS)))
def test_every_integral_export_the_station_does_not_implement_is_refused_at_build(low, i, tmp_path):
    make, why = REFUSALS[i]
    spec = make()
    name = f"bad{i:02d}"
    pdir = _export(tmp_path, spec, name)
    with pytest.raises(ValueError, match=why) as e:
        _build(low, pdir)
    msg = str(e.value)
    assert msg.startswith(f"{low}: policy "), msg
    assert name in msg and "refused" in msg, msg


@pytest.mark.parametrize("low,base", [("rl", POLICIES / "traj12_s2" / "obs_spec.json"),
                                      ("rl_7d", FIX / "rl7d_smoke_obs_spec.json")])
def test_the_axes_lows_refuse_an_integral_export(low, base, tmp_path):
    """rl / rl_7d: no axes / axes6 export with the integral exists and their observation is not parity-tested with it
    (OBS_INTEGRAL_LOW False) — a well-formed integral export is refused there with the reason."""
    s = _spec(base)
    n_act = int(s["num_actions"])
    s["frame_layout"] = [x for x in s["frame_layout"] if not x.startswith("prev_action")] + list(INTEGRAL_SLOTS) + \
        [f"prev_action({n_act})"]
    s["frame_dim"] = int(s["frame_dim"]) + 5
    s["obs_integral"] = _oi()
    assert check_frame_layout(s, "test", "x", goal_velocity=True if low == "rl_7d" else None,
                              n_actions=6 if low == "rl_7d" else None) is bool(s.get("obs_goal_frame", False))
    pdir = _export(tmp_path, s, "axes_int")
    with pytest.raises(ValueError, match=rf"LOW {low} does not fly it \(only rl_pwm / rl_pwm_7d") as e:
        _build(low, pdir)
    assert getattr(e.value, "export_refused", False) is True and "axes_int" in str(e.value)


def test_check_obs_integral_reads_the_law_and_defaults_only_the_round2_keys():
    s = _pi_spec()
    law = check_obs_integral(s, "t", "x", 0.05)
    assert law == {"tau_s": 4.0, "pos_max": 0.4, "att_max": 0.8, "att_error": "quat_xy", "cond_pos_m": 0.0,
                   "cond_att_rad": 0.0, "slots": list(INTEGRAL_SLOTS), "control_dt": 0.05, "offset_clip": 1.0,
                   "defaulted": ["att_error", "cond_pos_m", "cond_att_rad"], "law_round": 1,
                   "implementations": s["obs_integral"]["implementations"]}
    s2 = case_spec("r2_euler_rp_cond")
    law2 = check_obs_integral(s2, "t", "x", 0.05)
    assert law2["att_error"] == "euler_rp" and law2["cond_pos_m"] == 0.1 and law2["cond_att_rad"] == 0.3
    assert law2["defaulted"] == [] and law2["law_round"] == 2
    # explicit round-2 keys at the round-1 values: the round-1 text, nothing defaulted
    s3 = _pi_spec()
    s3["obs_integral"] = with_rlc_text({**s3["obs_integral"], "att_error": "quat_xy", "cond_pos_m": 0.0,
                                        "cond_att_rad": 0.0})
    law3 = check_obs_integral(s3, "t", "x", 0.05)
    assert law3["defaulted"] == [] and law3["law_round"] == 1
    assert s3["obs_integral"]["update_law"] == s["obs_integral"]["update_law"]      # explicit defaults: the round-1 text
    assert check_obs_integral(_plain_pwm_spec(), "t", "x", 0.05) is None                  # no block
    off = {**_plain_pwm_spec(), "obs_integral": _oi(enabled=False, slots=[])}
    assert check_obs_integral(off, "t", "x", 0.05) is None                                # round-1 P export
    off2 = {**_plain_pwm_spec(), "obs_integral": _oi(enabled=False, slots=[], att_error="euler_rp", cond_pos_m=0.0,
                                                    cond_att_rad=0.0)}
    assert check_obs_integral(off2, "t", "x", 0.05) is None                               # round-2 P export
    with pytest.raises(ValueError, match="does not exceed the control step 4"):
        check_obs_integral(s, "t", "x", 4.0)
    with pytest.raises(ValueError, match="does not exceed the control step nan"):
        check_obs_integral(s, "t", "x", float("nan"))


def test_the_integrator_refuses_what_it_cannot_run():
    with pytest.raises(ValueError, match="att_error"):
        StationErrorIntegrator(0.05, 4.0, 0.4, 0.8, 1.0, att_error="quat_xyz")
    for bad in ({"tau_s": 0.05}, {"pos_max": 0.0}, {"att_max": float("nan")}, {"offset_clip": -1.0},
                {"cond_pos_m": -0.1}, {"cond_att_rad": float("inf")}):
        kw = {"dt": 0.05, "tau_s": 4.0, "pos_max": 0.4, "att_max": 0.8, "offset_clip": 1.0, **bad}
        with pytest.raises(ValueError, match="StationErrorIntegrator"):
            StationErrorIntegrator(**kw)
    g = StationErrorIntegrator(0.05, 4.0, 0.4, 0.8, 1.0)
    assert g.leak == 1.0 - 0.05 / 4.0 and g.ticks == 0 and np.all(g.I_pos == 0.0) and np.all(g.I_att == 0.0)


# =============================================================================
# (e) old exports: bit-identical to the module before this change
# =============================================================================
N_LEGACY = 32
#: Follow-up 2026-10-18 (review T6): outside the recording's numerics environment (numerics_env) every float of the
#: legacy golden is compared to this absolute tolerance (rtol 0); the structure, integers (pulses, statuses, guard codes,
#: counts), strings, booleans and None stay exact everywhere. On the golden's STORED ticks this module differs
#: from it by at most 4.974e-14 (u: the rl LOW's wrench, N) and 7.772e-16 (obs) under numpy 1.26.4 + OpenBLAS 0.3.23
#: (env robust) against the recording's numpy 2.4.4 + OpenBLAS 0.3.31, with pulses, statuses, meta and the build log
#: equal [측정 RL_controller/logs/r2dev_20261018/station_port/followup/t6_drift_robust.txt]: 1e-12 is 20x that.
LEGACY_ATOL = 1e-12
#: legacy_ticks' run, as the golden stores it
LEGACY_INPUT_KEYS = ("t", "eta", "nu", "p_ref", "yaw_ref", "v_ref", "r_ref", "rp", "n_tags", "age", "feed", "resets")


def numerics_env() -> dict:
    """What decides the last bits of this module's float arithmetic on a given input (review T6): the numpy version (its
    ufunc kernels: np.sin / np.arctan2 / np.gradient move by an ulp between releases), the CPU features numpy dispatches
    them on, and the BLAS numpy was built with (the policy MLP's matmul). The legacy golden records it as
    meta.recorded_env; where it is the same here the golden is compared bit for bit, elsewhere to LEGACY_ATOL."""
    env = {"numpy": np.__version__, "machine": platform.machine(), "cpu_features": None, "blas": None}
    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            feats = importlib.import_module(name).__cpu_features__
        except (ImportError, AttributeError):
            continue
        env["cpu_features"] = sorted(k for k, v in feats.items() if v)
        break
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        env["blas"] = f"{blas.get('name')} {blas.get('version')}"
    except Exception:                                   # noqa: BLE001 - numpy < 1.25 has no mode="dicts"
        pass
    return env


def legacy_inputs_to_json(inp: dict) -> dict:
    """legacy_ticks' run as the golden stores it (JSON: a float64 round-trips exactly)."""
    out = {}
    for k in LEGACY_INPUT_KEYS:
        if k in ("n_tags", "resets"):
            out[k] = [int(v) for v in inp[k]]
        elif k == "feed":
            out[k] = [bool(v) for v in inp[k]]
        else:
            out[k] = np.asarray(inp[k], dtype=np.float64).tolist()
    return out


def legacy_inputs(stored: dict) -> dict:
    """The golden's stored ticks back in legacy_ticks' form (its dtypes)."""
    out = {k: np.asarray(stored[k], dtype=np.float64) for k in LEGACY_INPUT_KEYS if k not in ("n_tags", "feed", "resets")}
    out["n_tags"] = np.asarray(stored["n_tags"], dtype=int)
    out["feed"] = np.asarray(stored["feed"], dtype=bool)
    out["resets"] = [int(r) for r in stored["resets"]]
    return out


def assert_tree_close(got, want, atol: float, where: str = "") -> None:
    """``got`` == ``want`` for JSON-like trees (dict / list / str / int / float / bool / None): every float within
    ``atol`` (rtol 0; NaN == NaN, inf only == inf), everything else EXACT — the same keys, lengths and kinds (an int never
    matches a float, a bool never an int), integers, strings, booleans and None."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and sorted(got) == sorted(want), (where, got if not isinstance(got, dict) else
                                                                       sorted(set(got) ^ set(want)))
        for k in want:
            assert_tree_close(got[k], want[k], atol, f"{where}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (where, got, want)
        for i, (a, b) in enumerate(zip(got, want)):
            assert_tree_close(a, b, atol, f"{where}[{i}]")
    elif type(want) is float:
        assert type(got) is float, (where, got, want)
        assert got == want or (math.isnan(got) and math.isnan(want)) or abs(got - want) <= atol, (where, got, want)
    else:
        assert type(got) is type(want) and got == want, (where, got, want)


def legacy_ticks(seed: int, n: int = N_LEGACY) -> dict:
    """A run for the A/B: a smooth wander with noise at ~11 deg nose-up, goals inside and (ticks 8..13) beyond the
    leash, r_ref == 0 every 4th tick (the station's yaw-rate recovery), tilted plan references (the 7d LOWs fly
    them), a thin fix (1 tag) over ticks 5..12 (-> the tag gate HOLDs), a stale fix at 22..23, two coast ticks (3, 4:
    nothing left, note_applied_pwm(None)) and a re-engage (reset) at n // 2."""
    rng = np.random.default_rng(int(seed))
    k = np.arange(n)
    t = DT * k
    ph = rng.uniform(0.0, 2.0 * np.pi, 6)
    eta = np.stack([0.25 * np.sin(0.8 * t + ph[0]), 0.20 * np.sin(0.6 * t + ph[1]), 0.75 + 0.08 * np.sin(1.1 * t + ph[2]),
                    0.10 * np.sin(1.7 * t + ph[3]), 0.20 + 0.08 * np.sin(1.3 * t + ph[4]),
                    np.arctan2(np.sin(2.9 + 0.08 * k), np.cos(2.9 + 0.08 * k))], axis=1)
    eta = eta + rng.normal(0.0, 1.0, eta.shape) * np.array([0.004, 0.004, 0.004, 0.003, 0.003, 0.003])
    nu = np.gradient(eta, DT, axis=0) + rng.normal(0.0, 0.02, eta.shape)
    off = rng.uniform(-0.5, 0.5, (n, 3))
    off[8:14] = np.array([1.4, -0.9, 0.3])                    # beyond the 0.8 m leash
    r_ref = rng.uniform(-0.3, 0.3, n)
    r_ref[::4] = 0.0
    n_tags = np.full(n, 4, dtype=int)
    n_tags[5:13] = 1
    age = np.full(n, 0.05)
    age[22:24] = 0.5
    feed = np.ones(n, dtype=bool)
    feed[3:5] = False
    return {"t": t, "eta": eta, "nu": nu, "p_ref": eta[:, :3] + off,
            "yaw_ref": np.arctan2(np.sin(eta[:, 5] + 0.3), np.cos(eta[:, 5] + 0.3)),
            "v_ref": rng.uniform(-0.1, 0.1, (n, 3)), "r_ref": r_ref,
            "rp": np.c_[rng.uniform(-0.12, 0.12, n), rng.uniform(-0.24, 0.24, n)],
            "n_tags": n_tags, "age": age, "feed": feed, "resets": [0, n // 2]}


def drive_ab(ctrl, inp: dict, plan: bool) -> dict:
    """The whole tick, as the worker runs it: (re)engage, reference (a one-stage NedPlan carrying rp for the 7d LOWs),
    note_fix (pwm LOWs), step, the applied command fed back (pulses — None on a coast tick — or the pseudo-wrench).
    Records what the NETWORK was fed (policy.act wrapped), u, pulses, prev_action, last_action and info (solve_ms
    dropped: a timing)."""
    seen = []
    act = ctrl.policy.act

    def rec(obs):
        seen.append(np.array(obs, dtype=np.float64))
        return act(obs)

    ctrl.policy.act = rec
    pwm_low = getattr(ctrl, "transport", None) == "pwm"
    rs = {int(r) for r in inp["resets"]}
    out = {"u": [], "pwm": [], "prev": [], "last": [], "info": []}
    try:
        for k in range(len(inp["t"])):
            if k in rs:
                ctrl.reset()
            p, yaw = np.asarray(inp["p_ref"][k], float), float(inp["yaw_ref"][k])
            v, r = np.asarray(inp["v_ref"][k], float), float(inp["r_ref"][k])
            if plan:
                ctrl.set_path_plan_ned(NedPlan(p_ned=p.reshape(3, 1), yaw_ned=np.array([yaw]), v_ned=v.reshape(3, 1),
                                               r_ned=np.array([r]), psi_path=np.zeros(1),
                                               rp_ned=np.asarray(inp["rp"][k], float).reshape(2, 1)))
            else:
                ctrl.set_target_ned(p, yaw, v_ned=v, r_ned=r)
            if pwm_low:
                ctrl.note_fix(int(inp["n_tags"][k]), float(inp["age"][k]))
            u, info = ctrl.step(np.asarray(inp["eta"][k], float), np.asarray(inp["nu"][k], float), np.zeros(6),
                                float(inp["t"][k]))
            if pwm_low:
                ctrl.note_applied_pwm(info["pwm_us"] if bool(inp["feed"][k]) else None)
                out["pwm"].append([int(x) for x in info["pwm_us"]])
            else:
                ctrl.note_applied(u)
            out["u"].append(np.asarray(u, dtype=np.float64).tolist())
            out["prev"].append(np.asarray(ctrl._prev_action, dtype=np.float64).tolist())
            out["last"].append(np.asarray(ctrl.last_action, dtype=np.float64).tolist())
            out["info"].append(_canon({kk: vv for kk, vv in info.items() if kk != "solve_ms"}))
    finally:
        ctrl.policy.act = act
    out["obs"] = np.array(seen)
    return out


#: (LOW, installed export, seed) — driven against the frozen pre-change module; the first three (tracked exports)
#: are also pinned in LEGACY. pwm11att14R_s1 is skipped where it is not installed.
AB_RUNS = [("rl", "traj12_s2", 31), ("rl_pwm", "pwm11k0.2_s0", 32), ("rl_pwm_7d", "pwm11att7_s0", 33),
           ("rl", "traj5_s2", 34), ("rl", "traj10_s1", 35), ("rl_pwm", "pwm10_s1", 36),
           ("rl_pwm_7d", "att7pwm_s0", 37), ("rl_pwm_7d", "pwm11att9X_s0", 38), ("rl_pwm_7d", "pwm11att14R_s1", 39)]
LEGACY_RUNS = AB_RUNS[:3]


def _disabled_block(flavour: str) -> dict:
    """What a P export (enabled false) carries: the live (round-1) exporter's block, or the round-2 dev exporter's."""
    b = _oi(enabled=False, slots=[])
    if flavour == "round2":
        b.update({"att_error": "quat_xy", "cond_pos_m": 0.0, "cond_att_rad": 0.0})
    return b


def _ab_pair(low: str, pdir: Path, mod_old):
    lo_n, lo_o = [], []
    new = _build(low, pdir, log=lo_n.append)
    old = _build(low, pdir, mod=mod_old, log=lo_o.append)
    return new, old, lo_n, lo_o


def _assert_runs_identical(a: dict, b: dict, what) -> None:
    for k in ("obs",):
        assert a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes(), (what, k)
    for k in ("u", "pwm", "prev", "last", "info"):
        assert _dump(a[k]) == _dump(b[k]), (what, k)


@pytest.mark.parametrize("low,name,seed", AB_RUNS, ids=[f"{lo}-{n}" for lo, n, _ in AB_RUNS])
@pytest.mark.parametrize("flavour", ["no_block", "round1_off", "round2_off"])
def test_old_exports_are_bit_identical_to_the_frozen_pre_change_module(low, name, seed, flavour, tmp_path):
    old = _frozen()
    if old is None:
        pytest.skip(f"frozen pre-change copy not on this machine ({FROZEN}); the legacy golden still pins three exports")
    src = POLICIES / name
    if not (src / "obs_spec.json").is_file():
        pytest.skip(f"{name} not installed here")
    if flavour == "no_block":
        pdir = src
    else:                                   # the same export carrying a P export's obs_integral block (enabled false)
        spec = _spec(src / "obs_spec.json")
        spec["obs_integral"] = _disabled_block("round1" if flavour == "round1_off" else "round2")
        pdir = _export(tmp_path, spec, name, weights_from=src)
    new_c, old_c, log_n, log_o = _ab_pair(low, pdir, old)
    assert new_c._integ is None and new_c.obs_integral is None and not hasattr(old_c, "_integ")
    assert _norm_log(log_n) == _norm_log(log_o) and len(log_n) >= 1
    inp = legacy_ticks(seed)
    plan = low.endswith("_7d")
    a, b = drive_ab(new_c, inp, plan), drive_ab(old_c, inp, plan)
    _assert_runs_identical(a, b, (low, name, flavour))
    assert len(a["obs"]) == N_LEGACY
    if low != "rl":
        assert any(i.get("pwm_guard") == GUARD_HOLD for i in a["info"])          # the HOLD is inside the run
    # meta: byte-identical (REPO_ROOT is this checkout for both modules), no obs_integral key
    assert _dump(new_c.meta()) == _dump(old_c.meta())
    assert "obs_integral" not in new_c.meta()
    # observation() alone on a fresh history, the same calls
    for c in (new_c, old_c):
        c.reset()
    for k in range(8):
        for c in (new_c, old_c):
            c.set_target_ned(inp["p_ref"][k], float(inp["yaw_ref"][k]), v_ned=inp["v_ref"][k], r_ned=float(inp["r_ref"][k]))
        assert new_c.observation(inp["eta"][k], inp["nu"][k], float(inp["t"][k])).tobytes() == \
            old_c.observation(inp["eta"][k], inp["nu"][k], float(inp["t"][k])).tobytes()


def test_rl_7d_smoke_export_is_bit_identical_to_the_frozen_module(tmp_path):
    old = _frozen()
    if old is None:
        pytest.skip("frozen pre-change copy not on this machine")
    pdir = _export(tmp_path, _spec(FIX / "rl7d_smoke_obs_spec.json"), "rl7d")
    new_c, old_c, log_n, log_o = _ab_pair("rl_7d", pdir, old)
    assert _norm_log(log_n) == _norm_log(log_o)
    inp = legacy_ticks(40)
    a, b = drive_ab(new_c, inp, True), drive_ab(old_c, inp, True)
    _assert_runs_identical(a, b, "rl_7d")
    assert _dump(new_c.meta()) == _dump(old_c.meta())


def record_legacy_golden(mod, inputs: dict | None = None) -> dict:
    """The legacy golden: LEGACY_RUNS driven by ``mod`` (the FROZEN pre-change module — never the new one). ``inputs`` =
    run key -> its stored ticks (legacy_inputs_to_json); None = legacy_ticks(seed) drawn here. Every run stores the
    ticks it was driven with (follow-up 2026-10-18, review T6): the test replays them, never a re-draw."""
    runs = {}
    for low, name, seed in LEGACY_RUNS:
        key = f"{low}__{name}"
        stored = legacy_inputs_to_json(legacy_ticks(seed)) if inputs is None else inputs[key]
        log = []
        c = _build(low, POLICIES / name, mod=mod, log=log.append)
        d = drive_ab(c, legacy_inputs(stored), low.endswith("_7d"))
        fd = int(c.frame_dim)
        runs[key] = {"frames": d["obs"][:, :fd].tolist(), "u": d["u"], "pwm": d["pwm"],
                     "prev": d["prev"], "last": d["last"], "info": d["info"],
                     "meta": _canon(c.meta()), "log": _norm_log(log),
                     "export_sha256": {f: _sha256(POLICIES / name / f) for f in ("obs_spec.json", "policy_weights.npz")},
                     "inputs": stored}
    return runs


@pytest.mark.parametrize("low,name,seed", LEGACY_RUNS, ids=[f"{lo}-{n}" for lo, n, _ in LEGACY_RUNS])
@pytest.mark.parametrize("flavour", ["no_block", "round1_off", "round2_off"])
def test_old_exports_reproduce_the_legacy_golden(low, name, seed, flavour, tmp_path):
    """Always (no RL_controller needed): the new module on the frozen module's recorded runs — obs history, u, pulses,
    prev_action, last_action, info, meta and the build log — with no block, and with a P export's disabled block.

    Follow-up 2026-10-18 (review T6): replayed on the golden's STORED ticks (never a re-draw: legacy_ticks under another
    numpy differs in the last bits) and compared BIT FOR BIT wherever this is the recording's numerics environment
    (numerics_env() == meta.recorded_env); elsewhere every float to LEGACY_ATOL (rtol 0) with the structure, every
    integer (pulses, statuses, guard codes, counts), string (build log, guard reasons, paths), boolean and None still
    exact. On any numpy the bit-level proof stays the frozen-module A/B above (both modules in this process)."""
    g = json.loads(LEGACY.read_text(encoding="utf-8"))
    G = g["runs"][f"{low}__{name}"]
    src = POLICIES / name
    for f in ("obs_spec.json", "policy_weights.npz"):
        assert G["export_sha256"][f] == _sha256(src / f), f"{name}/{f} changed since the golden was recorded"
    assert g["meta"]["frozen_sha256"] == FROZEN_SHA256
    if flavour == "no_block":
        pdir = src
    else:
        spec = _spec(src / "obs_spec.json")
        spec["obs_integral"] = _disabled_block("round1" if flavour == "round1_off" else "round2")
        pdir = _export(tmp_path, spec, name, weights_from=src)
    log = []
    c = _build(low, pdir, log=log.append)
    inp = legacy_inputs(G["inputs"])
    d = drive_ab(c, inp, low.endswith("_7d"))
    want = history_from_frames(np.asarray(G["frames"], float), inp["resets"], c.K)
    m = _canon(c.meta())
    gm = dict(G["meta"])
    if flavour != "no_block":                          # the tmp copy's path is the only difference
        m.pop("policy_dir"), gm.pop("policy_dir")
        log = [x.replace(os.path.relpath(pdir, ROOT), os.path.relpath(src, ROOT)) for x in log]
    what = (low, name, flavour)
    # everywhere: the structure and every non-float exactly, every float to LEGACY_ATOL
    assert d["obs"].shape == want.shape, what
    np.testing.assert_allclose(d["obs"], want, rtol=0.0, atol=LEGACY_ATOL, err_msg=str(what))
    for k in ("u", "pwm", "prev", "last", "info"):
        assert_tree_close(_canon(d[k]), G[k], LEGACY_ATOL, f"{what} {k}")
    assert d["pwm"] == G["pwm"] and [i["status"] for i in d["info"]] == [i["status"] for i in G["info"]], what
    assert_tree_close(m, gm, LEGACY_ATOL, f"{what} meta")
    assert _norm_log(log) == G["log"], what
    # the recording's numerics environment: bit for bit, as recorded
    if g["meta"]["recorded_env"] == numerics_env():
        assert d["obs"].tobytes() == want.tobytes(), what
        for k in ("u", "pwm", "prev", "last", "info"):
            assert _dump(d[k]) == _dump(G[k]), (what, k)
        assert _dump(m) == _dump(gm), what


def test_the_legacy_golden_covers_the_hold_the_far_goal_and_the_reengage():
    g = json.loads(LEGACY.read_text(encoding="utf-8"))
    for key, G in g["runs"].items():
        assert len(G["frames"]) == N_LEGACY
        if not key.startswith("rl__"):
            assert sum(i["pwm_guard"] == GUARD_HOLD for i in G["info"]) >= 3, key
            assert len(G["pwm"]) == N_LEGACY
    assert g["meta"]["frozen_sha256"] == FROZEN_SHA256 and g["meta"]["recorded_with"] == "frozen pre-change module"


def test_the_legacy_golden_stores_the_ticks_it_was_recorded_on():
    """Follow-up 2026-10-18 (review T6): every golden run carries the ticks it was recorded on — legacy_ticks(seed) of
    the recording (bit for bit in the recording's numerics environment; elsewhere the draw itself moves by up to 1.4e-14
    [측정 RL_controller/logs/r2dev_20261018/station_port/verify_tests/t6_legacy_inputs_cross_env.txt], so there the floats
    to LEGACY_ATOL and the integer / boolean ticks exactly) — and they carry what the run is for: coast ticks 3, 4, a
    thin fix over 5..12 (-> the tag gate HOLDs), goals beyond the 0.8 m leash over 8..13, r_ref == 0 every 4th tick, a
    stale fix at 22..23 and the re-engage at N_LEGACY // 2."""
    g = json.loads(LEGACY.read_text(encoding="utf-8"))
    env = g["meta"]["recorded_env"]
    assert env["numpy"] == g["meta"]["numpy"] and set(env) == set(numerics_env())
    exact = env == numerics_env()
    for low, name, seed in LEGACY_RUNS:
        s = g["runs"][f"{low}__{name}"]["inputs"]
        assert sorted(s) == sorted(LEGACY_INPUT_KEYS), (low, sorted(s))
        got, mine = legacy_inputs(s), legacy_ticks(seed)
        assert got["resets"] == mine["resets"] == [0, N_LEGACY // 2]
        for k in LEGACY_INPUT_KEYS[:-1]:
            a, b = got[k], np.asarray(mine[k])
            assert a.shape == b.shape and a.dtype == b.dtype and len(a) == N_LEGACY, (low, k)
            if exact or a.dtype.kind in "ib":
                assert np.array_equal(a, b), (low, k)
            else:
                np.testing.assert_allclose(a, b, rtol=0.0, atol=LEGACY_ATOL, err_msg=f"{low} {k}")
        assert list(np.flatnonzero(~got["feed"])) == [3, 4]
        assert list(np.flatnonzero(got["n_tags"] < 2)) == list(range(5, 13))
        assert set(range(8, 14)) <= set(np.flatnonzero(np.linalg.norm(got["p_ref"] - got["eta"][:, :3], axis=1) > 0.8))
        assert np.all(got["r_ref"][::4] == 0.0) and np.all(got["r_ref"][1::4] != 0.0)
        assert list(np.flatnonzero(got["age"] > 0.3)) == [22, 23]


def test_the_existing_trim0_golden_still_holds():
    """fixtures/rl_pwm_trim0_golden_20261018.json (test_rl_pwm_heave_trim's, recorded before the heave trim) on this
    module: pulses, prev_action, status, guard, pwm_scale and the hold pulses of both pwm LOWs."""
    HT = importlib.import_module("rov_gui.tests.test_rl_pwm_heave_trim")
    g = json.loads(HT.GOLDEN.read_text(encoding="utf-8"))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for name, low, blk, own in HT.SCENARIOS:
            G = g["scenarios"][name]
            r = HT._build(low, blk, own, tmp)
            d = HT.drive(r, HT.scenario_ticks(HT.SEEDS[name]))
            assert d["pwm_us"] == G["pwm_us"], name
            assert np.array_equal(np.array(d["prev_action"]), np.array(G["prev_action"])), name
            assert d["status"] == G["status"] and d["guard"] == G["guard"], name
            assert np.max(np.abs(np.array(d["pwm_scale"]) - np.array(G["pwm_scale"]))) <= 1e-12, name
            assert [int(v) for v in r.hold_pwm_us] == G["hold_pwm_us"], name


def test_the_goal_frame_legacy_fixture_moves_identically_on_both_modules():
    """fixtures/rl_goalframe_legacy_obs.npz (test_rl_goal_frame's, recorded 2026-10-15; its meta says numpy 1.26.4):
    ALREADY BEFORE this change (baseline 2026-10-18) traj12_s2 deviates from it in obs and u and pwm11k0.2_s0 in u
    (obs and pulses match), by a few ulp [측정 RL_controller/logs/r2dev_20261018/station_port/
    goalframe_npz_check_20261018.txt, numpy 2.4.4] — pre-existing, not this port's. Pinned here: on those inputs this
    module moves bit for bit as the frozen pre-change one does, and the pulses still match the npz."""
    old = _frozen()
    if old is None:
        pytest.skip("frozen pre-change copy not on this machine")
    GF = importlib.import_module("rov_gui.tests.test_rl_goal_frame")
    d, _meta = GF._npz(GF.LEGACY)
    for name, low in (("traj12_s2", "rl"), ("pwm11k0.2_s0", "rl_pwm")):
        inp = GF._sub(d, f"{name}__")
        got = GF.drive_legacy(_build(low, POLICIES / name), inp)
        ref = GF.drive_legacy(_build(low, POLICIES / name, mod=old), inp)
        for k in ("obs", "u", "pwm"):
            assert got[k].shape == ref[k].shape and got[k].tobytes() == ref[k].tobytes(), (name, k)
        assert got["pwm"].tobytes() == inp["pwm"].tobytes(), name                # the pulses still match the npz


# =============================================================================
# (f) the real pretest PI export on the shipped config
# =============================================================================
@pytest.mark.parametrize("low", LOWS_PWM)
def test_the_real_pi_export_builds_and_flies_on_the_shipped_config(low, tmp_path):
    if not (PI_EXPORT / "obs_spec.json").is_file() or not (PI_EXPORT / "policy_weights.npz").is_file():
        pytest.skip(f"the pretest PI export is not on this machine ({PI_EXPORT})")
    pdir = tmp_path / "pretestPI"
    pdir.mkdir()
    for f in ("obs_spec.json", "policy_weights.npz"):
        shutil.copy(PI_EXPORT / f, pdir / f)
    assert _spec(pdir / "obs_spec.json") == _pi_spec()                       # the export the fixtures were made from
    cfg = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml")
    setattr(cfg, low, dict(getattr(cfg, low) or {}, policy_dir=str(pdir)))
    lines = []
    c = CLS[low](cfg, log=lines.append)
    assert c.obs_integral is not None and c.obs_integral["att_error"] == "quat_xy" and c.policy.obs_dim == 136
    assert sum("integral-error slots" in m for m in lines) == 2, lines             # the marker + the law line
    lo, hi = 1500 - 400 * c.pwm_cap - 1e-9, 1500 + 400 * c.pwm_cap + 1e-9
    rng = np.random.default_rng(5)
    c.reset()
    c.set_target_ned(np.array([0.10, -0.05, 0.80]), 0.3)
    for k in range(50):
        eta = np.array([0.02 * math.sin(0.3 * k), 0.01, 0.75 + 0.01 * math.cos(0.2 * k), 0.02, 0.15, 0.25])
        eta = eta + rng.normal(0.0, 0.002, 6)
        c.note_fix(4, 0.05)
        u, info = c.step(eta, rng.normal(0.0, 0.02, 6), np.zeros(6), k * DT)
        assert info["status"] == 0 and np.all(np.isfinite(u)), (k, info)
        assert all(isinstance(p, int) and lo <= p <= hi for p in info["pwm_us"]), info["pwm_us"]
        assert np.all(np.isfinite(c.last_action))
        c.note_applied_pwm(info["pwm_us"])
    m = c.meta()
    json.dumps(m)
    assert m["obs_integral"]["counts"]["ticks"] == 50 and m["obs_integral"]["counts"]["ticks_nonfinite"] == 0
    assert m["obs_integral"]["counts"]["ticks_add_masked"] == 0                    # its own pulses left every tick
    assert m["obs_integral"]["defaulted"] == ["att_error", "cond_pos_m", "cond_att_rad"]
    assert any("DEFAULTED: att_error, cond_pos_m, cond_att_rad" in x for x in lines), lines   # review F3, at the pool edge
    assert m["frame_layout"][-3:-1] == list(INTEGRAL_SLOTS)


# =============================================================================
# (g) the override rule (fix round 2026-10-18, review F1)
# =============================================================================
def _I_state(c) -> np.ndarray:
    """The integral state (I_pos world z-up, I_att) after the last update."""
    return np.r_[c._integ.I_pos, c._integ.I_att]


@pytest.mark.parametrize("low", LOWS_PWM)
def test_the_tag_gate_hold_masks_the_integral_add_and_keeps_the_leak(low, tmp_path):
    """Two followers on the same smooth states (the guard's jump reject and slew never act, so both see the same error
    every tick): A always has 4 tags, B a thin fix over ticks 6..25 (-> the tag gate HOLDs and the HOLD pulses leave).
    B's network is still stepped every tick. B's integral equals A's through the first HOLD tick (that update follows a
    command of the policy's own); every update from the next one through the first one after the HOLD follows HOLD
    pulses and is exactly the leak (nothing added), while A keeps adding; after it B adds exactly A's add again. So the
    HOLD can only SHRINK B's integral — the wind-up of review F1 is gone — and info / meta say which updates were masked."""
    pdir = _export(tmp_path, _pi_spec(), "pi", weights_from=PI_EXPORT if PI_EXPORT.is_dir() else None)
    A, B = _build(low, pdir), _build(low, pdir)
    n = 44
    seen = {"A": [], "B": []}
    for tag, c in (("A", A), ("B", B)):
        act = c.policy.act
        c.policy.act = (lambda a, s: (lambda o: (s.append(np.array(o, float)), a(o))[1]))(act, seen[tag])
        c.reset()
        c.set_target_ned(np.array([0.10, 0.05, 0.80]), 0.2)        # 4-5 cm off: the integral grows, far from its clamp
    guards, info_b, I = [], [], {"A": [], "B": []}
    for k in range(n):
        eta = np.array([0.06 + 0.001 * k, 0.0, 0.76, 0.01, 0.12, 0.2])
        nu = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
        for tag, c in (("A", A), ("B", B)):
            c.note_fix(1 if (tag == "B" and 6 <= k < 26) else 4, 0.05)
            u, info = c.step(eta, nu, np.zeros(6), k * DT)
            assert info["status"] == 0, (tag, k, info)
            if tag == "B":
                guards.append(info["pwm_guard"])
                info_b.append(info)
                if info["pwm_guard"] == GUARD_HOLD:
                    assert tuple(info["pwm_us"]) == tuple(int(v) for v in c.hold_pwm_us)
            else:
                assert info["pwm_guard"] != GUARD_HOLD and info["obs_int_add_masked"] is False
            c.note_applied_pwm(info["pwm_us"])
            I[tag].append(_I_state(c))
    hold = [k for k, gcode in enumerate(guards) if gcode == GUARD_HOLD]
    assert len(hold) >= 15 and hold == list(range(hold[0], hold[-1] + 1)), guards
    h0, h1 = hold[0], hold[-1]
    masked = set(range(h0 + 1, h1 + 2))                            # the updates that FOLLOW a HOLD tick
    IA, IB = np.array(I["A"]), np.array(I["B"])
    assert IA[:h0 + 1].tobytes() == IB[:h0 + 1].tobytes()          # identical through the first HOLD tick
    for k in range(h0 + 1, n):
        if k in masked:
            assert np.array_equal(IB[k], IB[k - 1] * LEAK), k       # the leak, nothing added (no clamp is near)
            assert np.abs(IA[k] - IA[k - 1] * LEAK).max() > 1e-4, k  # while A adds dt e
        else:                                                      # the same add as A's (the same error)
            assert np.abs((IB[k] - IB[k - 1] * LEAK) - (IA[k] - IA[k - 1] * LEAK)).max() <= 1e-15, k
    assert [i["obs_int_add_masked"] for i in info_b] == [k in masked for k in range(n)]
    assert np.all(np.abs(IB[h1 + 1]) < np.abs(IB[h0])) and np.abs(IA[h1 + 1, 1:3]).min() > np.abs(IA[h0, 1:3]).max()
    # the network saw B's slots = its (masked) state rotated, every tick, HOLD or not
    sb = np.array([o[SL_INT] for o in seen["B"]])
    assert len(sb) == n and [list(map(float, s)) for s in sb] == [i["obs_int_slots"] for i in info_b]
    mb, ma = B.meta()["obs_integral"]["counts"], A.meta()["obs_integral"]["counts"]
    assert mb["ticks"] == ma["ticks"] == n
    assert (mb["ticks_add_masked"], mb["overrides"], mb["override_ticks_max"]) == (len(masked), 1, len(masked))
    assert (ma["ticks_add_masked"], ma["overrides"], ma["override_ticks_max"]) == (0, 0, 0)


@pytest.mark.parametrize("low", LOWS_PWM)
def test_nothing_sent_masks_the_next_add_coast_grace_and_disengage(low, tmp_path):
    """The worker's station-bridge coast and pwm-gate grace window step the follower, send eight 1500s and tell it
    note_applied_pwm(None); a disengage does the same. Review F1: the update after each such tick is exactly the leak;
    one None masks exactly one update (the shortest grace); the next engage starts unmasked (one update from zero)."""
    c = _build(low, _export(tmp_path, _pi_spec(), "pi"))
    c.reset()
    target, eta = np.array([0.10, 0.05, 0.80]), np.array([0.06, 0.0, 0.76, 0.01, 0.12, 0.2])
    c.set_target_ned(target, 0.2)
    none_after = set(range(10, 20)) | {30}                         # a 10-tick coast, then a 1-tick grace
    I, flags = [], []
    for k in range(40):
        c.note_fix(4, 0.05)
        u, info = c.step(eta, np.zeros(6), np.zeros(6), k * DT)
        assert info["status"] == 0 and info["pwm_guard"] != GUARD_HOLD
        flags.append(info["obs_int_add_masked"])
        c.note_applied_pwm(None if k in none_after else info["pwm_us"])
        I.append(_I_state(c))
    want = {k + 1 for k in none_after}
    assert flags == [k in want for k in range(40)]
    I = np.array(I)
    for k in range(1, 40):
        if k in want:
            assert np.array_equal(I[k], I[k - 1] * LEAK), k
        else:
            assert np.abs(I[k] - I[k - 1] * LEAK).max() > 1e-4, k   # an add of dt e (|e| >= 4 cm)
    cnt = c.meta()["obs_integral"]["counts"]
    assert (cnt["ticks_add_masked"], cnt["overrides"], cnt["override_ticks_max"]) == (11, 2, 10)
    c.note_applied_pwm(None)                                       # the disengage's neutral ...
    c.reset()                                                      # ... and the next engage
    c.set_target_ned(target, 0.2)
    c.note_fix(4, 0.05)
    u, info = c.step(eta, np.zeros(6), np.zeros(6), 50 * DT)
    fr = c._hist[0].copy()                                          # the engage tick's frame
    assert info["obs_int_add_masked"] is False and info["obs_int_slots"] == [float(v) for v in fr[SL_INT]]
    assert np.abs(fr[SL_INT] - _one_update(c, fr, eta, target, _pi_spec()["obs_integral"])).max() <= 1e-12


@pytest.mark.parametrize("low", LOWS_PWM)
def test_a_three_second_thin_fix_hold_below_the_goal_no_longer_winds_up(low, tmp_path):
    """Review F1, case 1: 1 tag for 3 s with the hull 0.15 m below its goal, the HOLD pulses leaving. Integrated
    through (the 06:11 module) I_z reached 0.318 m*s, slot 0.795 [유도: the law on this input, RL_controller/logs/
    r2dev_20261018/station_port/verify_safety/f1_windup_out_20261018.txt] — an integral the policy was never allowed to
    act on; the pretest export's nominal heave goes from -3.3 to -40.0 N NED between slot 0 and 0.8 [유도: its nominal
    thrust model, same folder, f1_wrench_out_20261018.txt]. Under the override rule only the updates up to the first
    HOLD tick add; the rest leak."""
    c = _build(low, _export(tmp_path, _pi_spec(), "pi", weights_from=PI_EXPORT if PI_EXPORT.is_dir() else None))
    c.reset()
    c.set_target_ned(np.array([0.0, 0.0, 0.80]), 0.0)
    guards, slots = [], []
    for k in range(60):
        c.note_fix(1, 0.05)
        u, info = c.step(np.array([0.0, 0.0, 0.95, 0.0, 0.0, 0.0]), np.zeros(6), np.zeros(6), k * DT)
        assert info["status"] == 0
        guards.append(info["pwm_guard"])
        slots.append(info["obs_int_slots"])
        c.note_applied_pwm(info["pwm_us"])
    h0 = guards.index(GUARD_HOLD)
    assert guards[h0:] == [GUARD_HOLD] * (60 - h0) and h0 <= 4, guards
    C = np.diag([1.0, -1.0, -1.0])
    e_z = float((C @ np.array([0.0, 0.0, 0.80]))[2] - (C @ np.array([0.0, 0.0, 0.95]))[2])   # z-up goal - hull: +0.15 m
    i_through = i_rule = 0.0
    for k in range(60):
        i_through = min(i_through * LEAK + DT * e_z, 0.4)
        i_rule = i_rule * LEAK + (DT * e_z if k <= h0 else 0.0)
    assert i_through / 0.4 > 0.79                                  # the wind-up integrate-through gave (0.795)
    assert abs(c._integ.I_pos[2] - i_rule) <= 1e-12 and abs(slots[-1][2] - i_rule / 0.4) <= 1e-12
    assert slots[-1][2] < 0.05 and max(s[2] for s in slots) <= (h0 + 1) * DT * e_z / 0.4 + 1e-12
    assert c.meta()["obs_integral"]["counts"]["ticks_add_masked"] == 60 - h0 - 1


# =============================================================================
# the leash, non-finite ticks, meta and the build line
# =============================================================================
@pytest.mark.parametrize("d_up,want_e", [((2.0, 0.0, 0.0), (0.8, 0.0, 0.0)), ((3.0, 4.0, 0.0), (0.48, 0.64, 0.0)),
                                         ((0.3, -0.2, 0.1), (0.3, -0.2, 0.1))])
def test_e_pos_is_the_leashed_goal_offset(d_up, want_e, tmp_path):
    """The station convention: e_pos_w = clip(d, +-1) of the LEASHED d (leash_m 0.8, bearing kept) that offset_b uses;
    inside the leash that is training's clip(goal - p, +-1) exactly (training has no leash)."""
    c = _build("rl_pwm", _export(tmp_path, _pi_spec(), "pi"))
    eta = np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    c.reset()
    c.set_target_ned(eta[:3] + np.asarray(d_up) * np.array([1.0, -1.0, -1.0]), 0.0)
    fr = c.observation(eta, np.zeros(6), 0.0)[:34]
    want = DT * np.asarray(want_e) / 0.4                            # level, yaw 0: body = world z-up
    assert np.abs(fr[SL_POS] - want).max() <= 1e-15, (fr[SL_POS], want)
    assert np.abs(fr[4:7] - np.asarray(want_e)).max() <= 1e-15      # offset_b: the same leashed d
    m = c.meta()["obs_integral"]
    assert "LEASHED" in m["leash_note"] and "leash_m 0.8" in m["leash_note"]


@pytest.mark.parametrize("low", LOWS_PWM)
@pytest.mark.parametrize("bad", ["position_nan", "roll_nan", "goal_nan", "roll_inf"])
def test_a_non_finite_tick_is_refused_and_does_not_poison_the_integral(low, bad, tmp_path):
    """A non-finite state (or goal) on one tick: step returns the neutral pulses with a counted solver failure and
    never raises; that tick's integral update is NOT committed (ticks_nonfinite), so after the history has flushed the
    non-finite frame the follower flies again with the integral a twin that never saw the tick has. An infinite angle
    never reaches the integrator: math.cos(inf) raises in quat_from_euler_xyz first (the step's own guard answers it;
    no frame, no update — the next tick flies)."""
    pdir = _export(tmp_path, _pi_spec(), "pi")
    c, twin = _build(low, pdir), _build(low, pdir)
    eta = np.array([0.05, -0.02, 0.78, 0.02, 0.1, 0.3])
    nu = np.zeros(6)
    target = np.array([0.15, 0.05, 0.85])
    for x in (c, twin):
        x.reset()
        x.set_target_ned(target, 0.25)
    # fix round (review F1): c's failed ticks (the bad one, then K-1 more while the non-finite frame is in the history,
    # none for roll_inf) each mask the NEXT update's add — the twin is told "nothing left" after the same ticks, so the
    # two integrals must still end equal: the failures change nothing else
    n_bad = 0 if bad == "roll_inf" else c.K - 1          # the non-finite frame stays in the history K-1 more ticks
    twin_none_after = {5} | set(range(7, 7 + n_bad))
    for k in range(6):
        for x in (c, twin):
            x.note_fix(4, 0.05)
            assert x.step(eta, nu, np.zeros(6), k * DT)[1]["status"] == 0
            x.note_applied_pwm(None if (x is twin and k in twin_none_after) else NEUTRAL_PULSES)
    before = (c._integ.I_pos.copy(), c._integ.I_att.copy())
    e_bad = eta.copy()
    if bad == "position_nan":
        e_bad[1] = float("nan")
    elif bad == "roll_nan":
        e_bad[3] = float("nan")
    elif bad == "roll_inf":
        e_bad[3] = float("inf")
    else:
        c.set_target_ned(np.array([float("nan"), 0.0, 0.8]), 0.25)
    c.note_fix(4, 0.05)
    u, info = c.step(e_bad, nu, np.zeros(6), 6 * DT)
    assert info["status"] == 1 and tuple(info["pwm_us"]) == tuple(NEUTRAL_PULSES) and info["n_fail"] == 1
    assert np.array_equal(c._integ.I_pos, before[0]) and np.array_equal(c._integ.I_att, before[1])
    cnt = c.meta()["obs_integral"]["counts"]
    assert cnt["ticks_nonfinite"] == (0 if bad == "roll_inf" else 1) and cnt["ticks"] == 6, cnt
    c.set_target_ned(target, 0.25)
    c.note_applied_pwm(None)
    statuses, masked = [], []
    for k in range(7, 7 + 8):
        for x in (c, twin):
            x.note_fix(4, 0.05)
            u, info = x.step(eta, nu, np.zeros(6), k * DT)
            if x is c:
                statuses.append(info["status"])
                masked.append(info["obs_int_add_masked"])
            x.note_applied_pwm(None if (x is twin and k in twin_none_after) else NEUTRAL_PULSES)
    assert statuses == [1] * n_bad + [0] * (8 - n_bad), statuses
    assert masked == [True] * (1 + n_bad) + [False] * (7 - n_bad), masked     # tick 7 and one per failure after it
    assert np.array_equal(c._integ.I_pos, twin._integ.I_pos) and np.array_equal(c._integ.I_att, twin._integ.I_att)
    assert c.meta()["obs_integral"]["counts"]["ticks"] == twin.meta()["obs_integral"]["counts"]["ticks"] == 14
    assert c.meta()["obs_integral"]["counts"]["ticks_add_masked"] == \
        twin.meta()["obs_integral"]["counts"]["ticks_add_masked"] == 1 + n_bad


@pytest.mark.parametrize("low", LOWS_PWM)
def test_meta_and_build_line_of_an_integral_export(low, tmp_path):
    spec = case_spec("r2_euler_rp_cond")
    lines = []
    c = _build(low, _export(tmp_path, spec, "pi"), log=lines.append)
    ints = [m for m in lines if m.startswith(f"{low}: integral-error slots")]
    assert len(ints) == 1, lines
    assert "e_att euler_rp" in ints[0] and "|e| < 0.1 m / 0.3 rad" in ints[0] and "NOT in the CSV" in ints[0]
    assert "round-2 law text" in ints[0] and "MASKED (leak only)" in ints[0] and "DEFAULTED" not in ints[0]
    assert "integral-error slots" in lines[0]                       # the LOW's own build line carries the marker
    m = c.meta()["obs_integral"]
    for k, v in (("enabled", True), ("tau_s", 4.0), ("pos_max", 0.4), ("att_max", 0.8), ("att_error", "euler_rp"),
                 ("cond_pos_m", 0.1), ("cond_att_rad", 0.3), ("slots", list(INTEGRAL_SLOTS)), ("control_dt", 0.05),
                 ("offset_clip_m", 1.0), ("defaulted", []), ("law_round", 2)):
        assert m[k] == v, k
    assert "StationErrorIntegrator" in m["implementation"] and "LEASHED" in m["leash_note"]
    assert m["export_implementations"] == spec["obs_integral"]["implementations"]
    assert "NOT in the CSV" in m["note"] and "obs_int_slots" in m["per_tick"]
    assert "STATION convention" in m["override_rule"] and "note_applied_pwm(None)" in m["override_rule"]
    assert m["counts"] == {"ticks": 0, "ticks_any_saturated": 0, "ticks_nonfinite": 0, "ticks_add_masked": 0,
                           "overrides": 0, "override_ticks_max": 0, "max_abs_slot": [0.0] * 5, "last_slots": None}
    assert m["engagement"] == 1 and m["previous_engagements"] == [] and m["previous_engagements_dropped"] == 0
    json.dumps(c.meta())
    c.reset()
    c.set_target_ned(np.array([0.05, 0.0, 0.82]), 0.0)
    for k in range(5):
        c.observation(np.array([0.0, 0.0, 0.8, 0.0, 0.05, 0.0]), np.zeros(6), k * DT)
    m = c.meta()["obs_integral"]
    assert m["counts"]["ticks"] == 5 and m["counts"]["last_slots"] is not None
    assert m["counts"]["max_abs_slot"] == [abs(v) for v in m["counts"]["max_abs_slot"]]
    c.reset()                                                     # a new engagement: fresh counts and state
    m = c.meta()["obs_integral"]
    assert m["counts"]["ticks"] == 0 and m["state"] == {"I_pos_w": [0.0] * 3, "I_att_rp": [0.0] * 2}
    assert m["engagement"] == 2 and [p["counts"]["ticks"] for p in m["previous_engagements"]] == [5]   # review F2
    # the world-frame twin flies too (obs_goal_frame false + the integral)
    w = _build(low, _export(tmp_path, case_spec("r1_world_frame"), "piw"))
    assert w.goal_frame is False and w._integ is not None


# =============================================================================
# (h) the per-tick record and the engagement records (fix round 2026-10-18, review F2)
# =============================================================================
@pytest.mark.parametrize("low", LOWS_PWM)
def test_step_info_carries_the_slots_the_network_saw(low, tmp_path):
    """Per tick, info obs_int_slots = the 5 integral slots of the newest frame the network was fed and obs_int_add_masked
    = whether the override rule masked that update (None on a tick that built no frame) — for a CSV column workers.py
    does not have yet. Only an integral export carries the keys: an installed export of before has neither (the A/B
    tests pin the rest of its info byte for byte)."""
    c = _build(low, _export(tmp_path, _pi_spec(), "pi"))
    seen = []
    act = c.policy.act
    c.policy.act = lambda o: (seen.append(np.array(o, float)), act(o))[1]
    c.reset()
    c.set_target_ned(np.array([0.2, -0.1, 0.9]), 0.4)
    eta = np.array([0.05, 0.02, 0.8, 0.03, 0.1, 0.3])
    flags = []
    for k in range(14):
        c.note_fix(1 if 3 <= k < 9 else 4, 0.05)                  # a thin fix -> a HOLD in the middle
        u, info = c.step(eta, np.zeros(6), np.zeros(6), k * DT)
        assert info["status"] == 0 and info["obs_int_slots"] == [float(v) for v in seen[-1][SL_INT]], k
        flags.append(info["obs_int_add_masked"])
        c.note_applied_pwm(info["pwm_us"])
    assert all(isinstance(f, bool) for f in flags) and any(flags) and not all(flags)
    c.note_fix(4, 0.05)
    u, info = c.step(np.array([0.05, 0.02, 0.8, float("inf"), 0.1, 0.3]), np.zeros(6), np.zeros(6), 14 * DT)
    assert info["status"] == 1 and info["obs_int_slots"] is None and info["obs_int_add_masked"] is None
    name = "pwm11k0.2_s0" if low == "rl_pwm" else "pwm11att7_s0"
    old = _build(low, POLICIES / name)
    old.reset()
    old.set_target_ned(np.array([0.2, -0.1, 0.9]), 0.4)
    old.note_fix(4, 0.05)
    _, info_old = old.step(eta, np.zeros(6), np.zeros(6), 0.0)
    assert old._integ is None and not any(str(k).startswith("obs_int") for k in info_old), sorted(info_old)


@pytest.mark.parametrize("low", LOWS_PWM)
def test_a_reengage_keeps_the_finished_engagements_record(low, tmp_path):
    """reset() at each engage starts fresh counts, so the meta a REC CSV writes when it closes over two engagements used
    to describe only the last. The finished engagement's record — its index, the t handed to step (CSV t_traj) and the
    wall-clock span, its counts and state — now stays in previous_engagements (the last OBS_INTEGRAL_ENGAGEMENTS_KEPT,
    the rest counted as dropped); a reset without a tick adds none."""
    c = _build(low, _export(tmp_path, _pi_spec(), "pi"))
    c.reset()                                                      # engage 1 (the build's own reset recorded nothing)
    c.set_target_ned(np.array([0.0, 0.0, 0.40]), 0.0)
    for k in range(200):                                           # 0.55 m below the goal for 10 s: I_z saturates
        c.note_fix(4, 0.05)
        _, info = c.step(np.array([0.0, 0.0, 0.95, 0.0, 0.0, 0.0]), np.zeros(6), np.zeros(6), 100.0 + k * DT)
        c.note_applied_pwm(info["pwm_us"])
    first = c.meta()["obs_integral"]
    assert first["engagement"] == 1 and first["previous_engagements"] == []
    assert first["t_traj_s"] == [100.0, 100.0 + 199 * DT] and first["wall_s"][0] <= first["wall_s"][1]
    c.note_applied_pwm(None)                                       # the disengage
    c.reset()                                                      # engage 2
    c.set_target_ned(np.array([0.0, 0.0, 0.94]), 0.0)
    for k in range(10):
        c.note_fix(4, 0.05)
        _, info = c.step(np.array([0.0, 0.0, 0.95, 0.0, 0.0, 0.0]), np.zeros(6), np.zeros(6), 5.0 + k * DT)
        c.note_applied_pwm(info["pwm_us"])
    m = c.meta()["obs_integral"]
    assert m["engagement"] == 2 and m["counts"]["ticks"] == 10 and m["t_traj_s"] == [5.0, 5.0 + 9 * DT]
    (p,) = m["previous_engagements"]
    assert p["engagement"] == 1 and p["counts"] == first["counts"] and p["state"] == first["state"]
    assert p["t_traj_s"] == first["t_traj_s"] and p["wall_s"] == first["wall_s"] and p["wall_s"][1] <= m["wall_s"][0]
    assert p["counts"]["ticks"] == 200 and p["counts"]["ticks_any_saturated"] > 100
    assert p["counts"]["max_abs_slot"][2] == 1.0 and m["counts"]["max_abs_slot"][2] < 0.05
    json.dumps(c.meta())
    c.reset()
    c.reset()                                                      # a reset without a tick: no record
    m = c.meta()["obs_integral"]
    assert [r["engagement"] for r in m["previous_engagements"]] == [1, 2] and m["engagement"] == 3
    for _ in range(OBS_INTEGRAL_ENGAGEMENTS_KEPT + 5):             # many short engagements: bounded, counted
        c.reset()
        c.observation(np.array([0.0, 0.0, 0.95, 0.0, 0.0, 0.0]), np.zeros(6), 0.0)
    m = c.meta()["obs_integral"]
    assert len(m["previous_engagements"]) == OBS_INTEGRAL_ENGAGEMENTS_KEPT
    assert m["previous_engagements_dropped"] == (2 + OBS_INTEGRAL_ENGAGEMENTS_KEPT + 4) - OBS_INTEGRAL_ENGAGEMENTS_KEPT
    assert m["previous_engagements"][-1]["engagement"] == m["engagement"] - 1


# =============================================================================
# (i) the law texts (review F4) and the defaulted round-2 keys (review F3)
# =============================================================================
def test_the_law_texts_equal_the_recorded_rl_texts():
    """rl_policy.obs_integral_law_text / OBS_INTEGRAL_UNITS_TEXT against fixtures/rl_integral_law_texts_20261018.json
    (recorded from RL_controller's generators: the live tree's round-1 text and the r2dev copy's round-2 clauses, on a
    grid of constants and options) and against the real pretest export — on any machine."""
    fx = _law_texts()
    assert len(fx["texts"]) >= 12 and {r["att_error"] for r in fx["texts"]} == {"quat_xy", "euler_rp"}
    for r in fx["texts"]:
        got = obs_integral_law_text(r["tau_s"], r["pos_max"], r["att_max"], r["att_error"], r["cond_pos_m"],
                                    r["cond_att_rad"])
        assert got == r["dev"], r
        if r["live"] is not None:
            assert got == r["live"], r
        assert (" ROUND-2 OPTIONS: " in got) == (r["att_error"] == "euler_rp" or r["cond_pos_m"] > 0 or
                                                 r["cond_att_rad"] > 0)
    assert fx["units"]["live"] == fx["units"]["dev"] == OBS_INTEGRAL_UNITS_TEXT
    oi = _pi_spec()["obs_integral"]
    assert oi["update_law"] == obs_integral_law_text(4.0, 0.4, 0.8) and oi["units"] == OBS_INTEGRAL_UNITS_TEXT
    with pytest.raises(ValueError, match="att_error"):
        obs_integral_law_text(4.0, 0.4, 0.8, "euler_xyz")


def test_the_law_texts_equal_the_rl_generators_live():
    """The same against RL_controller's sources themselves (exec'd / parsed from their text, read-only), on a wider grid:
    a change of the RL text generators shows here at the next station test run, before an export carrying it is
    refused at the pool."""
    if not (RLC_LIVE_OBS_SPEC.is_file() and RLC_DEV_OBS_SPEC.is_file()):
        pytest.skip("RL_controller rov_rl/obs_spec.py not on this machine (the recorded texts still pin it)")
    live = _load_source(RLC_LIVE_OBS_SPEC, "rlc_obs_spec_live_text")
    dev = _load_source(RLC_DEV_OBS_SPEC, "rlc_obs_spec_dev_text")
    n = 0
    for tau in (4.0, 2.5, 10.0, 0.75):
        for pm, am in ((0.4, 0.8), (0.25, 0.5), (1.0, 0.5)):
            for ae in ("quat_xy", "euler_rp"):
                for cp, ca in ((0.0, 0.0), (0.1, 0.3), (0.1, 0.0), (0.0, 0.3), (0.05, 0.123456)):
                    want = dev.integral_update_law(tau, pm, am, ae, cp, ca)
                    assert obs_integral_law_text(tau, pm, am, ae, cp, ca) == want, (tau, pm, am, ae, cp, ca)
                    if ae == "quat_xy" and cp == 0.0 and ca == 0.0:
                        assert live.integral_update_law(tau, pm, am) == want
                    n += 1
    assert n == 120
    for p in (RLC_LIVE_EXPORTER, RLC_DEV_EXPORTER):
        if p.is_file():
            assert _exporter_units(p) == OBS_INTEGRAL_UNITS_TEXT, p


@pytest.mark.parametrize("low", LOWS_PWM)
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_every_rl_text_of_an_implemented_law_builds(low, case, tmp_path):
    """The other side of the text gate: every law the station implements, with the text RL_controller writes for it,
    builds on both pwm LOWs and reports its round (round 2 only with a ROUND-2 OPTIONS clause)."""
    c = _build(low, _export(tmp_path, case_spec(case), "pi"))
    want = 2 if CASES[case][0].get("att_error") == "euler_rp" or CASES[case][0].get("cond_pos_m", 0.0) > 0 else 1
    assert c.obs_integral["law_round"] == want and c.meta()["obs_integral"]["law_round"] == want


def test_the_build_line_says_which_round2_keys_were_defaulted(tmp_path):
    """Review F3: a block without att_error / cond_* is read as the round-1 law — and the live RL_controller exporter
    writes none of them even for a round-2 run — so the law line says so at build, where the operator reads it; an export
    that carries the keys (here: explicitly at the round-1 values) gets no such note."""
    lines = []
    _build("rl_pwm", _export(tmp_path, _pi_spec(), "r1"), log=lines.append)
    (law,) = [x for x in lines if x.startswith("rl_pwm: integral-error slots")]
    assert ("e_att quat_xy (DEFAULTED: att_error, cond_pos_m, cond_att_rad absent from the export, read as the round-1 "
            "law; a round-2 run exported by the live RL_controller exporter looks exactly like this") in law
    assert "round-1 law text" in law
    spec = _pi_spec()
    spec["obs_integral"] = with_rlc_text({**spec["obs_integral"], "att_error": "quat_xy", "cond_pos_m": 0.0,
                                          "cond_att_rad": 0.0})
    lines2 = []
    _build("rl_pwm_7d", _export(tmp_path, spec, "r2"), log=lines2.append)
    (law2,) = [x for x in lines2 if x.startswith("rl_pwm_7d: integral-error slots")]
    assert "DEFAULTED" not in law2 and "e_att quat_xy, dt 0.05 s" in law2


# =============================================================================
# the leash beyond its radius (review F5)
# =============================================================================
def test_beyond_the_leash_a_minor_axis_integrates_less_than_training(tmp_path):
    """Beyond the leash the station integrates the LEASHED offset d_i * leash_m / |d| where training integrates
    clip(d_i, +-1). The major axis reaches the clamp in both; a minor axis need not: for d = (3.0, 0, 0.3) m (z-up) the
    station's e_z is 0.3 * 0.8 / |d| = 0.0796 m, so after 20 s I_z = tau e_z (1 - leak^400) = 0.316 m*s (slot 0.79)
    where training's law clamps at 0.4 (slot 1.0) [유도]. |e_station| <= |e_training| per axis: the station errs small."""
    c = _build("rl_pwm", _export(tmp_path, _pi_spec(), "pi"))
    eta = np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    d_up = np.array([3.0, 0.0, 0.3])
    c.reset()
    c.set_target_ned(eta[:3] + d_up * np.array([1.0, -1.0, -1.0]), 0.0)
    i_train = np.zeros(3)
    for k in range(400):
        fr = c.observation(eta, np.zeros(6), k * DT)[:34]
        i_train = np.clip(i_train * LEAK + DT * np.clip(d_up, -1.0, 1.0), -0.4, 0.4)
    e_z = 0.3 * 0.8 / float(np.linalg.norm(d_up))
    assert abs(e_z - 0.0796) < 1e-4
    assert fr[SL_POS][0] == 1.0 and i_train[0] == 0.4                          # the major axis: clamped in both
    assert abs(fr[SL_POS][2] - 4.0 * e_z * (1.0 - LEAK ** 400) / 0.4) <= 1e-9   # the station's minor axis ...
    assert fr[SL_POS][2] < 0.8 and i_train[2] / 0.4 == 1.0                     # ... below the clamp training's reaches
    assert np.all(np.abs(fr[4:7]) <= np.abs(np.clip(d_up, -1.0, 1.0)) + 1e-12)  # offset_b: the same leashed d


# =============================================================================
# (j) follow-up 2026-10-18: the round-1 review findings the fix round did not get (tests T1-T5, parity F1-F3)
# =============================================================================
Q_ID = np.array([1.0, 0.0, 0.0, 0.0])
OBS_INTEGRAL_ATT_ERRORS_HERE = ("quat_xy", "euler_rp")


def _rlc_integrator(att_error: str, **kw):
    """RL_controller's numpy integrator twin — the r2dev copy's deploy/obs_builder.ErrorIntegratorNp (round 1 + the
    round-2 options; exec'd from source, read-only) — or None when it is not on this machine."""
    ob = _rlc_builder("dev")
    return None if ob is None else ob.ErrorIntegratorNp(att_err=att_error, **kw)


def _station_and_twin(att_error: str, cond_pos: float = 0.0, cond_att: float = 0.0, clip: float = 1.0) -> tuple:
    kw = dict(dt=DT, tau_s=4.0, pos_max=0.4, att_max=0.8, offset_clip=clip, cond_pos_m=cond_pos, cond_att_rad=cond_att)
    return StationErrorIntegrator(att_error=att_error, **kw), _rlc_integrator(att_error, **kw)


def _update_both(g, twin, e_src, q_des, q_est=Q_ID) -> None:
    """One update of the station's integrator and (when here) of the dev twin on the same error; their states must be
    equal bit for bit (the same operations in the same order)."""
    g.update(np.asarray(e_src, float), q_des, q_est, q_est)
    if twin is not None:
        twin.update(np.asarray(e_src, float), np.zeros(3), q_des, q_est)
        assert np.array_equal(twin.I_pos, g.I_pos) and np.array_equal(twin.I_att, g.I_att), \
            (twin.I_pos, g.I_pos, twin.I_att, g.I_att)


def _q_axes(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) Ry(pitch) Rx(roll) as a (w,x,y,z) quaternion, composed here from the three axis rotations (independent of
    rl_policy.quat_from_euler_xyz)."""
    qx = np.array([math.cos(0.5 * roll), math.sin(0.5 * roll), 0.0, 0.0])
    qy = np.array([math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0])
    qz = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
    return _hamilton(_hamilton(qz, qy), qx)


# ----------------------------------------------------------------------------- T1: the GUARDED position
@pytest.mark.parametrize("low", LOWS_PWM)
def test_the_integral_takes_the_guarded_position_not_a_rejected_fix_jump(low, tmp_path):
    """Review T1 (medium): the integral is updated with the GUARDED state — the position after PwmObsGuard's fix-jump
    reject — as training applies obs_guard before integ.update (RL_controller rov_env.py) and the HwRlPwm docstring
    promises. A, B and C fly one PI export through step() on the same ticks. Over ticks 10..12 the fix of A and B is thin
    (1 tag < min_tags 2; three ticks stay under the tag gate's 0.2 s, so no HOLD) and A's position jumps by
    (0.20, -0.15, 0.25) m NED (0.35 m > jump_m 0.15): A's guard holds the previous fix, so every network input A is fed —
    the integral slots included — and its integral state equal B's (which never jumped) byte for byte. C gets A's jump
    with a full fix (nothing rejected): at the first jump tick its int_pos_b moves against A's by exactly
    dt |jump| / pos_max (0.044) and int_att_rp not at all — what integrating the raw position would do, so the
    equality above is not vacuous."""
    pdir = _export(tmp_path, _pi_spec(), "pi", weights_from=PI_EXPORT if PI_EXPORT.is_dir() else None)
    fol = {k: _build(low, pdir) for k in "ABC"}
    seen = {k: [] for k in fol}
    for k, c in fol.items():
        act = c.policy.act
        c.policy.act = (lambda a, s: (lambda o: (s.append(np.array(o, float)), a(o))[1]))(act, seen[k])
        c.reset()
        c.set_target_ned(np.array([0.02, -0.03, 0.70]), -0.4)
    base = np.array([0.0, 0.0, 0.75, 0.02, 0.08, -0.4])
    jump = np.array([0.20, -0.15, 0.25])
    thin, n = {10, 11, 12}, 30
    guards, I = {k: [] for k in fol}, {k: [] for k in fol}
    for t in range(n):
        for k, c in fol.items():
            eta = base.copy()
            if k in "AC" and t in thin:
                eta[:3] += jump
            c.note_fix(1 if (k in "AB" and t in thin) else 4, 0.05)
            u, info = c.step(eta, np.zeros(6), np.zeros(6), t * DT)
            assert info["status"] == 0 and info["obs_int_add_masked"] is False, (k, t, info)
            guards[k].append(info["pwm_guard"])
            c.note_applied_pwm(info["pwm_us"])
            I[k].append(_I_state(c))
    assert guards["A"] == [GUARD_JUMP if t in thin else GUARD_NONE for t in range(n)], guards["A"]
    assert guards["B"] == guards["C"] == [GUARD_NONE] * n
    assert [fol[k].obs_guard.counts["jump"] for k in "ABC"] == [3, 0, 0]
    sa, sb, sc = (np.array(seen[k]) for k in "ABC")
    assert sa.shape == (n, 136) and sa.tobytes() == sb.tobytes()           # every input, the integral slots included
    assert np.array(I["A"]).tobytes() == np.array(I["B"]).tobytes()        # and the integral state itself
    t0 = min(thin)
    assert sa[:t0].tobytes() == sc[:t0].tobytes()
    dpos = float(np.linalg.norm(sc[t0, SL_POS] - sa[t0, SL_POS]))
    assert abs(dpos - DT * float(np.linalg.norm(jump)) / 0.4) <= 1e-12, dpos  # one raw tick integrated
    assert np.array_equal(sc[t0, SL_ATT], sa[t0, SL_ATT])


# ----------------------------------------------------------------------------- T2: the e_pos clamp
@pytest.mark.parametrize("low", LOWS_PWM)
@pytest.mark.parametrize("case", ["leash_2m", "clip_0.5"])
def test_e_pos_is_clipped_to_obs_offset_clip_where_the_leash_does_not_bound_it(low, case, tmp_path):
    """Review T2: e_pos_w = clip(d, +-obs_offset_clip) per world axis, as trained (clamp(goal - p_obs, +-clip)). The
    clamp acts only where the station's leash leaves a component above the clip: leash_m 2.0 (legal: rl_policy reads it
    with no upper bound) with the shipped clip 1.0 and a goal d = (1.5, -1.2, 0.3) m z-up (|d| 1.94 m, inside that
    leash) -> e_pos (1.0, -1.0, 0.3); or an export with obs_offset_clip 0.5 under the shipped leash 0.8 and
    d = (0.7, 0.3, -0.2) m -> e_pos (0.5, 0.3, -0.2). For 20 ticks the slots equal the law integrated here from first
    principles (level, heading 0: body = world z-up), I <- clip(I (1 - dt/tau) + dt e, +-pos_max), until the clipped
    axis sits in its clamp (slot exactly 1); offset_b shows the same clipped d."""
    spec, over = _pi_spec(), {}
    if case == "leash_2m":
        over = {"leash_m": 2.0}
        d_up, clip, leash = np.array([1.5, -1.2, 0.3]), 1.0, 2.0
    else:
        spec["obs_offset_clip"] = 0.5
        d_up, clip, leash = np.array([0.7, 0.3, -0.2]), 0.5, 0.8
    c = _build(low, _export(tmp_path, spec, "pi"), **over)
    assert c.leash_m == leash and c.clip_m == clip and c._integ.clip == clip
    assert np.linalg.norm(d_up) < leash and np.abs(d_up).max() > clip      # the leash keeps d, the clamp cuts it
    eta = np.array([0.1, -0.2, 0.8, 0.0, 0.0, 0.0])
    c.reset()
    c.set_target_ned(eta[:3] + d_up * np.array([1.0, -1.0, -1.0]), 0.0)
    e = np.clip(d_up, -clip, clip)
    i = np.zeros(3)
    for k in range(20):
        fr = c.observation(eta, np.zeros(6), k * DT)[:34]
        i = np.clip(i * LEAK + DT * e, -0.4, 0.4)
        assert np.abs(fr[SL_POS] - i / 0.4).max() <= 1e-12, (case, k, fr[SL_POS], i / 0.4)
        assert np.abs(fr[4:7] - e).max() <= 1e-12, (case, k, fr[4:7])           # offset_b: the same clip
    assert fr[21] == 1.0                                                       # the clipped axis in its clamp


def test_the_cond_mask_judges_the_clipped_e_pos():
    """Review T2 (its M57): the conditional mask is judged on the CLIPPED e_pos, as trained (ErrorIntegratorNp clips, then
    masks). It only shows when cond_pos_m exceeds obs_offset_clip (legal: check_obs_integral bounds neither): with clip
    1.0, cond 1.2 m and a source error (1.5, 1.1, -0.5) m every clipped |e| (1.0, 1.0, 0.5) is below 1.2, so all three
    are added — judged on the source, x would be masked. Against the dev ErrorIntegratorNp when it is here."""
    for att_error in OBS_INTEGRAL_ATT_ERRORS_HERE:
        g, twin = _station_and_twin(att_error, cond_pos=1.2, cond_att=0.3, clip=1.0)
        _update_both(g, twin, [1.5, 1.1, -0.5], Q_ID)
        assert np.array_equal(g.I_pos, DT * np.array([1.0, 1.0, -0.5])) and np.array_equal(g.I_att, np.zeros(2))


# ----------------------------------------------------------------------------- T3 / parity F1: the STRICT mask
#: (cond_pos_m, cond_att_rad): the pair this file's round-2 exports carry, and a dyadic pair
COND_EDGES = [(0.1, 0.3), (0.25, 0.125)]


@pytest.mark.parametrize("att_error", OBS_INTEGRAL_ATT_ERRORS_HERE)
@pytest.mark.parametrize("cond_pos,cond_att", COND_EDGES, ids=["cond0.1_0.3", "cond0.25_0.125"])
def test_the_conditional_mask_is_strict_at_its_threshold(att_error, cond_pos, cond_att):
    """Review T3 / parity F1: the round-2 mask adds dt e only where |e| < cond — STRICT, as trained (the dev copy's
    rov_rl/obs_integral.py and deploy/obs_builder.ErrorIntegratorNp) — per component: |e| == cond exactly -> nothing
    added there, one ulp below -> dt e added, one ulp above -> nothing; every position axis and both attitude components,
    both signs, every other component at zero error. Exact attitude errors: quat_xy with q_est = identity and
    q_des = (w, e/2, 0, 0) gives e_att.x = 2 (e/2) = e bit for bit; euler_rp (atan2 / asin are not exact) puts cond_att on
    the |e| this integrator computes instead (above it / below it by one ulp). Against the dev ErrorIntegratorNp."""
    up, down = (lambda x: float(np.nextafter(x, np.inf))), (lambda x: float(np.nextafter(x, 0.0)))
    n_added = n_masked = 0
    for edge, added in (("at", False), ("below", True), ("above", False)):
        for sign in (1.0, -1.0):
            for ax in range(3):                                           # position
                mag = {"at": cond_pos, "below": down(cond_pos), "above": up(cond_pos)}[edge]
                e = np.zeros(3)
                e[ax] = sign * mag
                g, twin = _station_and_twin(att_error, cond_pos, cond_att)
                _update_both(g, twin, e, Q_ID)
                assert np.array_equal(g.I_pos, DT * e if added else np.zeros(3)), (edge, sign, ax, g.I_pos)
                assert np.array_equal(g.I_att, np.zeros(2))
                n_added, n_masked = n_added + added, n_masked + (not added)
            for comp in range(2):                                         # attitude (roll, pitch)
                if att_error == "quat_xy":
                    ca = cond_att
                    h = 0.5 * sign * {"at": ca, "below": down(ca), "above": up(ca)}[edge]
                    q_des = np.array([math.sqrt(1.0 - h * h), 0.0, 0.0, 0.0])
                    q_des[1 + comp] = h
                    e_c = 2.0 * h
                else:
                    a = sign * cond_att
                    q_des = _q_axes(a, 0.0, 0.0) if comp == 0 else _q_axes(0.0, a, 0.0)
                    e_c = float(StationErrorIntegrator(DT, 4.0, 0.4, 0.8, 1.0, att_error="euler_rp").error(
                        np.zeros(3), q_des, Q_ID)[1][comp])
                    ca = {"at": abs(e_c), "below": up(abs(e_c)), "above": down(abs(e_c))}[edge]
                g, twin = _station_and_twin(att_error, cond_pos, ca)
                e_att = g.error(np.zeros(3), q_des, Q_ID)[1]
                assert e_att[comp] == e_c and e_att[1 - comp] == 0.0, (edge, e_att, e_c)
                assert (abs(e_c) == ca) == (edge == "at") and (abs(e_c) < ca) == added
                _update_both(g, twin, np.zeros(3), q_des)
                want = np.zeros(2)
                want[comp] = DT * e_c if added else 0.0
                assert np.array_equal(g.I_att, want) and np.array_equal(g.I_pos, np.zeros(3)), (edge, comp, g.I_att)
                n_added, n_masked = n_added + added, n_masked + (not added)
    assert (n_added, n_masked) == (10, 20)


@pytest.mark.parametrize("low", LOWS_PWM)
@pytest.mark.parametrize("edge_axis", [0, 1, 2])
def test_the_conditional_mask_is_strict_through_the_follower(low, edge_axis, tmp_path):
    """Review T3 through HwRlPwm / HwRlPwm7d.observation, on the round-2 export quat_xy + cond 0.1 m / 0.3 rad (with
    RL_controller's text): the hull at the datum (p = 0, level, heading 0 — the leashed d is the goal itself, bit for bit)
    and a goal EXACTLY 0.1 m (the float 0.1) off on one axis, one ulp below 0.1 m off on the next and 0.05 m on the third:
    the first frame's int_pos_b is exactly 0 on the first axis and exactly dt e / pos_max on the other two."""
    spec = case_spec("r2_quat_xy_cond")
    assert spec["obs_integral"]["cond_pos_m"] == 0.1
    c = _build(low, _export(tmp_path, spec, "pi"))
    d = np.zeros(3)
    d[edge_axis] = 0.1 if edge_axis != 2 else -0.1
    d[(edge_axis + 1) % 3] = -float(np.nextafter(0.1, 0.0))
    d[(edge_axis + 2) % 3] = 0.05
    c.reset()
    c.set_target_ned(d * np.array([1.0, -1.0, -1.0]), 0.0)
    fr = c.observation(np.zeros(6), np.zeros(6), 0.0)[:34]
    assert np.array_equal(fr[4:7], d)                                          # offset_b: the goal itself, exactly
    want = np.where(np.arange(3) == edge_axis, 0.0, DT * d) / 0.4
    assert np.array_equal(fr[SL_POS], want) and np.array_equal(fr[SL_ATT], np.zeros(2)), (fr[SL_INT], want)
    assert fr[21 + edge_axis] == 0.0 and np.all(fr[SL_POS][np.arange(3) != edge_axis] != 0.0)


# ----------------------------------------------------------------------------- T4 / parity F2: the euler_rp roll wrap
#: (roll_des, roll_est) [deg]: across +-180 deg (the wrap acts) and one pair that needs none
WRAP_CASES = [(170.0, -170.0), (-170.0, 170.0), (math.degrees(0.1), math.degrees(-3.1)), (179.0, -179.5), (10.0, -20.0)]


@pytest.mark.parametrize("roll_des_deg,roll_est_deg", WRAP_CASES, ids=[f"{a:+.1f}_{b:+.1f}" for a, b in WRAP_CASES])
def test_the_euler_rp_roll_difference_takes_the_short_way_round(roll_des_deg, roll_est_deg, tmp_path):
    """Review T4 / parity F2: att_error euler_rp integrates wrap(roll(q_des) - roll(q_est)) = atan2(sin, cos), as the dev
    copy does, so a goal and an estimate on opposite sides of +-180 deg give the SHORT way round: +170 / -170 deg ->
    -20 deg (unwrapped: +340 deg, the long way, the opposite sign). Integrator level with pitch and heading on both
    quaternions (the pitch slot = dt (0.05 - -0.02)), against the dev ErrorIntegratorNp when it is here; then through
    HwRlPwm7d.observation with the flown roll reference set by hand past the trained envelope (observation does not clip
    it; on the vehicle the tilt interlock and the trained envelope keep the wrap out of reach [스펙 config/hw_mpc.yaml
    engage.per_thruster.abort_deg 20; the PI export's attitude_reference.roll_max_rad 0.14]) — the goal-frame rotation
    and the w >= 0 flip leave ZYX roll as it is."""
    r_d, r_e = math.radians(roll_des_deg), math.radians(roll_est_deg)
    want = math.atan2(math.sin(r_d - r_e), math.cos(r_d - r_e))
    if abs(r_d - r_e) > math.pi:
        assert abs(want) < math.pi and math.copysign(1.0, want) != math.copysign(1.0, r_d - r_e)
    g, twin = _station_and_twin("euler_rp")
    q_d, q_e = _q_axes(r_d, 0.05, 0.3), _q_axes(r_e, -0.02, 1.1)
    _update_both(g, twin, np.zeros(3), q_d, q_e)
    assert abs(g.I_att[0] - DT * want) <= 1e-12 and abs(g.I_att[1] - DT * 0.07) <= 1e-12, (g.I_att, DT * want)
    c = _build("rl_pwm_7d", _export(tmp_path, case_spec("r2_euler_rp"), "pi"))
    c.reset()
    c.set_target_ned(np.array([0.0, 0.0, 0.8]), 0.3)
    c._rp_flown = (r_d, 0.0)
    fr = c.observation(np.array([0.0, 0.0, 0.8, r_e, 0.0, -0.4]), np.zeros(6), 0.0)[:34]
    assert abs(fr[24] - DT * want / 0.8) <= 1e-12 and abs(fr[25]) <= 1e-12, (fr[SL_ATT], DT * want / 0.8)


# ----------------------------------------------------------------------------- parity F3: what step() feeds
STEP_RESETS = (0, 24, 44)
N_STEP = 60


def step_inputs(tilted: bool, n: int = N_STEP, resets=STEP_RESETS) -> dict:
    """A short smooth run for the step()-level checks (parity F3): three engagements and no guard event (a full fix
    every tick, |d nu| per tick far below the slew's 0.1 m/s, no position jump), goals within 0.75 m (inside the leash),
    errors that move the integral (the first engagement's x drives I_x into its clamp and every |e_pos| > 0.1 m there,
    the second's stay below 0.08 m — both sides of the round-2 cond 0.1 m; heading errors up to ~150 deg), a
    reference yaw rate in [0.05, 0.35] rad/s (no recovery, no clip) and — tilted (the 7d LOW) — plan roll/pitch
    references that sit beyond the trained envelope and step faster than the trained goal rate, so the flown reference
    is clipped AND rate-limited. ``v_w`` = the world velocity (z-up) the RL builder is given; ``nu`` = a body velocity for
    the station-only check."""
    k = np.arange(n)
    t = DT * k
    eng = np.searchsorted(np.asarray(resets), k, side="right") - 1
    yaw = np.arctan2(np.sin(2.5 + 0.9 * t), np.cos(2.5 + 0.9 * t))
    eta = np.stack([0.25 * np.sin(0.8 * t + 0.3), 0.2 * np.sin(0.6 * t + 1.1), 0.8 + 0.1 * np.sin(1.1 * t),
                    0.15 * np.sin(1.7 * t + 0.4), 0.1 + 0.15 * np.sin(1.3 * t + 2.0), yaw], axis=1)
    off = np.array([[0.35, -0.30, 0.25], [-0.03, 0.04, -0.02], [0.20, 0.25, -0.30]])[eng]
    amp = np.array([0.1, 0.04, 0.1])[eng][:, None]
    d_up = off + amp * np.stack([np.sin(0.5 * t), np.cos(0.4 * t), np.sin(0.9 * t)], axis=1)
    psi = yaw + np.array([2.6, 0.2, -1.4])[eng] + 0.3 * np.sin(0.7 * t)
    rates = np.stack([0.10 * np.sin(1.3 * t), -0.08 * np.cos(0.9 * t), 0.2 * np.sin(0.6 * t)], axis=1)
    rp = np.zeros((n, 2))
    if tilted:
        rp = np.array([[0.30, -0.40], [0.05, 0.10], [-0.20, 0.30]])[eng] + 0.03 * np.stack([np.sin(t), np.cos(t)], axis=1)
    return {"t": t, "eta": eta, "p_ref": eta[:, :3] + d_up * np.array([1.0, -1.0, -1.0]),
            "yaw_ref": np.arctan2(np.sin(psi), np.cos(psi)),
            "v_ref": 0.08 * np.stack([np.cos(0.3 * t), np.sin(0.4 * t), 0.2 * np.cos(t)], axis=1),
            "r_ref": 0.2 + 0.15 * np.sin(0.35 * t), "rp": rp, "rates": rates,
            "v_w": np.stack([0.15 * np.sin(0.5 * t), 0.10 * np.cos(0.7 * t), 0.03 * np.sin(t)], axis=1),
            "nu": np.c_[0.12 * np.sin(0.5 * t), -0.1 * np.cos(0.7 * t), 0.03 * np.sin(t), rates],
            "resets": [int(r) for r in resets]}


def _flown_rp(prev: tuple, plan_rp, ar: dict) -> tuple:
    """The 7d LOWs' flown roll/pitch reference, recomputed here (parity F3): the plan's, clipped to the export's
    trained envelope (attitude_reference roll/pitch_max_rad), then moved toward by at most rate_max_rad_s * dt per axis
    from the previous tick's (level at engage)."""
    st = float(ar["rate_max_rad_s"]) * DT
    tgt = (float(np.clip(plan_rp[0], -ar["roll_max_rad"], ar["roll_max_rad"])),
           float(np.clip(plan_rp[1], -ar["pitch_max_rad"], ar["pitch_max_rad"])))
    return tuple(p + float(np.clip(g - p, -st, st)) for p, g in zip(prev, tgt))


def _set_reference(c, inp: dict, k: int, tilted: bool) -> None:
    """The tick's reference as the worker hands it: a one-stage NedPlan carrying the roll/pitch reference for the 7d
    LOW, a target for the level one."""
    p, yaw = np.asarray(inp["p_ref"][k], float), float(inp["yaw_ref"][k])
    v, r = np.asarray(inp["v_ref"][k], float), float(inp["r_ref"][k])
    if tilted:
        c.set_path_plan_ned(NedPlan(p_ned=p.reshape(3, 1), yaw_ned=np.array([yaw]), v_ned=v.reshape(3, 1),
                                    r_ned=np.array([r]), psi_path=np.zeros(1),
                                    rp_ned=np.asarray(inp["rp"][k], float).reshape(2, 1)))
    else:
        c.set_target_ned(p, yaw, v_ned=v, r_ned=r)


def _assert_integral_in_play(case: str, sl: np.ndarray) -> None:
    """The step runs' newest-frame slots (n, 5): the integral moved; without the cond mask it neared its clamps, with it
    the first engagement (every |e_pos| > 0.1 m) added nothing to I_pos while the second did."""
    assert np.abs(sl[:, 0:3]).max() > 0.0 and np.abs(sl[:, 3:5]).max() > 0.0, case
    if CASES[case][0].get("cond_pos_m", 0.0) == 0.0:
        assert np.abs(sl).max() > 0.5, case
    else:
        assert np.all(sl[:STEP_RESETS[1], 0:3] == 0.0) and np.any(sl[STEP_RESETS[1]:STEP_RESETS[2], 0:3] != 0.0), case


def _record_net(c) -> list:
    seen = []
    act = c.policy.act
    c.policy.act = lambda o: (seen.append(np.array(o, float)), act(o))[1]
    return seen


@pytest.mark.parametrize("low", LOWS_PWM)
@pytest.mark.parametrize("case", ["r1_quat_xy", "r2_euler_rp_cond"])
def test_step_feeds_the_network_what_observation_builds_on_the_flown_reference(low, case, tmp_path):
    """Parity F3, on any machine: the observation the network receives INSIDE HwRlPwm / HwRlPwm7d.step — after the
    observation guard, on the 7d LOW's envelope-clipped, rate-limited roll/pitch reference (_advance_rp_flown) and with
    the pulses fed back (note_applied_pwm) — is byte for byte what observation() builds on the same state with that
    flown reference and that previous action, tick for tick over three engagements without a guard event (asserted).
    observation() is what (a) pins to RL_controller's builders, so step() carries that parity: one integral update per
    tick, the flown q_des in the frame and in the integral, no mask on a tick the policy flew. The flown reference is
    recomputed here (_flown_rp) and must be the follower's; on the 7d run it is clipped and rate-limited."""
    spec = case_spec(case)
    pdir = _export(tmp_path, spec, "pi", weights_from=PI_EXPORT if PI_EXPORT.is_dir() else None)
    S, O = _build(low, pdir), _build(low, pdir)
    seen = _record_net(S)
    tilted = low == "rl_pwm_7d"
    inp = step_inputs(tilted)
    rp = (0.0, 0.0)
    for k in range(N_STEP):
        if k in STEP_RESETS:
            S.reset()
            O.reset()
            rp = (0.0, 0.0)
        for c in (S, O):
            _set_reference(c, inp, k, tilted)
        if tilted:
            rp = _flown_rp(rp, inp["rp"][k], spec["attitude_reference"])
        prev = S._prev_action.copy()
        S.note_fix(4, 0.05)
        u, info = S.step(inp["eta"][k], inp["nu"][k], np.zeros(6), float(inp["t"][k]))
        assert info["status"] == 0 and info["pwm_guard"] == GUARD_NONE and info["obs_int_add_masked"] is False, (k, info)
        S.note_applied_pwm(info["pwm_us"])
        if tilted:
            assert abs(S._rp_flown[0] - rp[0]) <= 1e-15 and abs(S._rp_flown[1] - rp[1]) <= 1e-15, (k, S._rp_flown, rp)
            O._rp_flown = rp
        O._prev_action = prev
        want = O.observation(inp["eta"][k], inp["nu"][k], float(inp["t"][k]))
        assert len(seen) == k + 1 and seen[-1].tobytes() == want.tobytes(), (k, float(np.abs(seen[-1] - want).max()))
    assert S.meta()["obs_integral"]["counts"]["ticks"] == N_STEP - STEP_RESETS[-1]
    _assert_integral_in_play(case, np.array(seen)[:, SL_INT])
    if tilted:
        assert S._rp_clip_ticks > 0 and S._rp_rate_ticks > 0                  # the last engagement: clipped and slewed


@pytest.mark.parametrize("case,low", RUNS, ids=[f"{c}-{lo}" for c, lo in RUNS])
def test_step_feeds_the_network_the_rl_builders_observation(case, low, tmp_path):
    """Parity F3 against RL_controller's builders themselves (exec'd from source, read-only; skipped where they are not):
    the observation the network receives inside step() equals deploy/obs_builder.ObsBuilder's history (<= TOL) tick for
    tick over the smooth three-engagement run — the builder fed what training sees: the tick's pose and IMU, its world
    velocity (the station gets the builder's body velocity back in FRD), the goal with the FLOWN roll/pitch (recomputed
    here, _flown_rp) and the previous action the station fed back (prev_action is a pass-through slot). No guard event
    (asserted): the guard's own effect on the integral is T1's."""
    which = ["live", "dev"] if CASES[case][2] == "live" else ["dev"]
    builders = [(w, _rlc_builder(w)) for w in which if _rlc_builder(w) is not None]
    if not builders:
        pytest.skip("RL_controller deploy/obs_builder.py not on this machine (the station-only step check still runs)")
    spec = case_spec(case)
    tilted = low == "rl_pwm_7d"
    inp = step_inputs(tilted)
    for w, ob in builders:
        S = _build(low, _export(tmp_path, spec, f"pi_{w}", weights_from=PI_EXPORT if PI_EXPORT.is_dir() else None))
        seen = _record_net(S)
        b = ob.ObsBuilder(spec)
        rp, worst = (0.0, 0.0), 0.0
        for k in range(N_STEP):
            t, eta, rates = float(inp["t"][k]), np.asarray(inp["eta"][k], float), np.asarray(inp["rates"][k], float)
            if k in STEP_RESETS:
                S.reset()
                b.reset()
                rp = (0.0, 0.0)
            if tilted:
                rp = _flown_rp(rp, inp["rp"][k], spec["attitude_reference"])
            b.set_goal(np.asarray(inp["p_ref"][k], float), float(inp["yaw_ref"][k]),
                       v_goal_ned=np.asarray(inp["v_ref"][k], float), yaw_rate_ned=float(inp["r_ref"][k]),
                       roll_ned=rp[0], pitch_ned=rp[1])
            b.fix_p, b.fix_yaw, b.fix_t = ob.C_NED_TO_UP @ eta[:3], -float(eta[5]), t
            b.v_w = np.asarray(inp["v_w"][k], float)
            b.step(t, None, {"roll": float(eta[3]), "pitch": float(eta[4]), "p": float(rates[0]), "q": float(rates[1]),
                             "r": float(rates[2])}, prev_action=S._prev_action.copy())
            v_b = b.last["v_b"]
            _set_reference(S, inp, k, tilted)
            S.note_fix(4, 0.05)
            u, info = S.step(eta, np.r_[v_b[0], -v_b[1], -v_b[2], rates], np.zeros(6), t)
            assert info["status"] == 0 and info["pwm_guard"] == GUARD_NONE and info["obs_int_add_masked"] is False, \
                (w, k, info)
            S.note_applied_pwm(info["pwm_us"])
            if tilted:
                assert abs(S._rp_flown[0] - rp[0]) <= 1e-15 and abs(S._rp_flown[1] - rp[1]) <= 1e-15, (w, k)
            assert len(seen) == k + 1
            worst = max(worst, float(np.abs(seen[-1] - np.asarray(b.hist, float).reshape(-1)).max()))
        assert worst <= TOL, (case, low, w, worst)
        _assert_integral_in_play(case, np.array(seen)[:, SL_INT])


# ============================================================================================== (g') a mask set between reset() and the first frame
# 2026-10-18 follow-up verifier (RL_controller/logs/r2dev_20261018/station_port/followup_verify/test_vf_killers.py): the engage refill
# must clear the override mask itself, not rely on reset()'s clear — a "nothing sent" report (coast / grace) or a tick that fails before
# any frame is built can arrive AFTER reset() and before the first frame. The trained law: the first frame already carries ONE dt e.
_K_TARGET, _K_ETA = np.array([0.10, 0.05, 0.80]), np.array([0.06, 0.0, 0.76, 0.01, 0.12, 0.2])


def _first_frame_is_one_unmasked_update(c, info):
    fr = c._hist[0].copy()
    assert info["status"] == 0, info
    assert info["obs_int_add_masked"] is False, info["obs_int_add_masked"]
    one = _one_update(c, fr, _K_ETA, _K_TARGET, _pi_spec()["obs_integral"])
    assert np.abs(one).max() > 1e-3                                       # a real add (|e| >= 4 cm)
    assert np.abs(fr[SL_INT] - one).max() <= 1e-12, (fr[SL_INT], one)
    assert c.meta()["obs_integral"]["counts"]["ticks_add_masked"] == 0


@pytest.mark.parametrize("low", LOWS_PWM)
def test_nothing_sent_between_reset_and_the_first_frame_does_not_mask_the_engage_update(low, tmp_path):
    c = _build(low, _export(tmp_path, _pi_spec(), "pi"))
    c.reset()
    c.set_target_ned(_K_TARGET, 0.2)
    c.note_applied_pwm(None)                       # nothing left after the engage's reset, before its first frame
    c.note_fix(4, 0.05)
    u, info = c.step(_K_ETA, np.zeros(6), np.zeros(6), 0.0)
    _first_frame_is_one_unmasked_update(c, info)


@pytest.mark.parametrize("low", LOWS_PWM)
def test_a_tick_failing_before_the_first_frame_does_not_mask_the_engage_update(low, tmp_path):
    c = _build(low, _export(tmp_path, _pi_spec(), "pi"))
    c.reset()
    c.set_target_ned(_K_TARGET, 0.2)
    c.note_fix(4, 0.05)
    u, info = c.step(_K_ETA[:3], np.zeros(6), np.zeros(6), 0.0)    # 3-float state: _frame raises before any frame
    assert info["status"] == 1 and c._hist is None and info["obs_int_slots"] is None, info
    c.note_applied_pwm(info["pwm_us"])
    c.note_fix(4, 0.05)
    u, info = c.step(_K_ETA, np.zeros(6), np.zeros(6), DT)
    _first_frame_is_one_unmasked_update(c, info)
