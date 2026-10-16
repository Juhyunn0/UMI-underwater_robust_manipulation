#!/usr/bin/env python3
"""Goal-frame observations for the RL LOWs (2026-10-15).

RL_controller exports with obs_spec ``obs_goal_frame: true`` (pwm11att7 and later) were trained with q_des and q_est
rotated by -goal yaw (the ZYX yaw of q_des) BEFORE the w >= 0 canonicalization (rov_env._get_observations; its deploy
twin deploy/obs_builder.py), and their frame_layout names the two slots q_des_goalframe / q_est_goalframe so that a
station without the transform refuses them. HwRl._frame builds that observation for every RL LOW (rl, rl_7d, rl_pwm,
rl_pwm_7d) behind one layout gate (rl_policy.check_frame_layout).

Pinned here:
  (i)   the gate, on all four LOWs: the goal-frame layout ONLY with obs_goal_frame true, the world-frame layout ONLY with
        it false / absent; a non-boolean flag, a mixed / truncated / missing layout, a frame_dim that is not the
        layout's width -> refused at build, the message naming the LOW, the export and what is wrong; between the two
        twins of one export ONLY the two quaternion slots differ;
  (ii)  parity with RL_controller deploy/obs_builder.py for the att7 export (pwm11att7_s0) over random states, goal
        headings including 179.9 / 180 / -179.9 deg and tilted goals, single frames AND the newest-first history:
        against fixtures/rl_goalframe_att7_parity.npz (that builder's own numbers) always, and against the builder
        itself (imported by path, read-only) when the RL_controller checkout is on this machine;
  (iii) continuity across +-180 deg (goal heading and vehicle heading), where the world-frame quaternions flip sign;
  (iv)  the installed world-frame exports build BYTE-identical observations, pseudo-wrenches and pulses to the module
        before this change: traj12_s2 on rl and pwm11k0.2_s0 on rl_pwm against fixtures/rl_goalframe_legacy_obs.npz
        always, and every installed export on its LOW against the 2026-10-15 pre-change copy of rl_policy.py
        (RL_controller logs/eval/station_backup_20261015) when it is on this machine.

Fixture provenance: RL_controller logs/eval/station_fix_20261015/impl-goalframe/scripts/make_goalframe_fixtures.py
(it imports the drivers below, so the fixtures and these tests make the same calls; each npz carries a ``meta`` JSON).

Run: QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest rov_gui/tests/test_rl_goal_frame.py -q
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rov_gui.control.geometry import MpcConfig
from rov_gui.control.path_geometry import NedPlan
from rov_gui.control.rl_policy import (HwRl, HwRl7d, HwRlPwm, HwRlPwm7d, Q_SLOTS_GOAL, Q_SLOTS_WORLD, _unique,
                                       check_frame_layout, frame_layout, goal_frame_quats, quat_from_euler_xyz,
                                       yaw_from_q)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIX = HERE / "fixtures"
POLICIES = ROOT / "rov_gui" / "control" / "rl_policies"
#: RL_controller logs/rsl_rl/rov_posehold/2026-10-15_02-01-35_pwm11att7_s0/exported/obs_spec.json, verbatim
ATT7_SPEC = FIX / "rl_goalframe_pwm11att7_s0_obs_spec.json"
PARITY = FIX / "rl_goalframe_att7_parity.npz"
LEGACY = FIX / "rl_goalframe_legacy_obs.npz"
#: read-only, and only when present on this machine
RLC = Path("/home/bdml/Desktop/RL_controller")
RLC_BUILDER = RLC / "deploy" / "obs_builder.py"
RLC_OBS_SPEC = RLC / "rov_rl" / "obs_spec.py"
BACKUP = RLC / "logs" / "eval" / "station_backup_20261015" / "rov_gui" / "control" / "rl_policy.py"

LOWS = ("rl", "rl_7d", "rl_pwm", "rl_pwm_7d")
CLS = {"rl": HwRl, "rl_7d": HwRl7d, "rl_pwm": HwRlPwm, "rl_pwm_7d": HwRlPwm7d}
#: the world-frame export each LOW is checked with (installed, or the exporter's rl_7d smoke spec)
WORLD_BASE = {"rl": POLICIES / "traj12_s2" / "obs_spec.json", "rl_7d": FIX / "rl7d_smoke_obs_spec.json",
              "rl_pwm": POLICIES / "pwm11k0.2_s0" / "obs_spec.json",
              "rl_pwm_7d": POLICIES / "att7pwm_s0" / "obs_spec.json"}
#: frame slices: q_des 0:4, offset_b 4:7, q_est 7:11, everything after it 11:
QD, OFF, QE, REST = slice(0, 4), slice(4, 7), slice(7, 11), slice(11, None)
_DROP = object()


# =============================================================================
# helpers
# =============================================================================
def _spec(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _goal_twin(spec: dict) -> dict:
    """The world-frame export's goal-frame twin, as RL_controller's exporter writes it: the two quaternion slots renamed
    and obs_goal_frame true, nothing else (spelled out here, not taken from rl_policy.frame_layout)."""
    s = json.loads(json.dumps(spec))
    ren = {"q_des(w,x,y,z)": "q_des_goalframe(w,x,y,z)", "q_est(w,x,y,z)": "q_est_goalframe(w,x,y,z)"}
    s["frame_layout"] = [ren.get(x, x) for x in s["frame_layout"]]
    s["obs_goal_frame"] = True
    return s


def _export(tmp, base: dict, name: str = "export", **over) -> Path:
    """A FORMAT export: ``base`` with ``over`` applied (_DROP removes a key) and a zero-weight MLP of base's shape — never
    a policy; the observation does not read the weights."""
    spec = json.loads(json.dumps(base))
    for k, v in over.items():
        if v is _DROP:
            spec.pop(k, None)
        else:
            spec[k] = v
    pdir = Path(tmp) / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "obs_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    n_in, n_act = int(base["frame_dim"]) * int(base["history_len"]), int(base["num_actions"])
    np.savez(pdir / "policy_weights.npz", n_layers=np.array(2), activation=np.array("elu"),
             W0=np.zeros((8, n_in), np.float32), b0=np.zeros(8, np.float32),
             W1=np.zeros((n_act, 8), np.float32), b1=np.linspace(-0.3, 0.3, n_act).astype(np.float32))
    return pdir


def _cfg(low: str, pdir, **block) -> MpcConfig:
    """MpcConfig flying ``pdir`` on ``low``. The pwm transport block: cap 1.0 (min(cap, trained) flies, so never below
    any export's trained cap) and the sign-probe gate opted out (offline test; test_rl_pwm_scale.py owns both)."""
    cfg = MpcConfig()
    cfg.rl_pwm = {"pwm_cap": 1.0, "require_thruster_probe": False}
    if low == "rl_pwm":
        cfg.rl_pwm.update({"policy_dir": str(pdir), **block})
    else:
        setattr(cfg, low, {"policy_dir": str(pdir), **block})
    return cfg


def _build(low: str, pdir, cls=None, **block):
    return (cls or CLS[low])(_cfg(low, pdir, **block), log=lambda m: None)


def _wrap(a):
    return (np.asarray(a, float) + math.pi) % (2.0 * math.pi) - math.pi


def _R(q) -> np.ndarray:
    """Rotation matrix of a unit (w,x,y,z) quaternion — written out here, independent of the module's quaternion code."""
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _load_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def _rlc_builder():
    """RL_controller deploy/obs_builder.py, imported by path (read-only), or None when the checkout is not here."""
    return _load_by_path(RLC_BUILDER, "rlc_deploy_obs_builder") if RLC_BUILDER.is_file() else None


@functools.lru_cache(maxsize=None)
def _backup_module():
    """The 2026-10-15 pre-change rl_policy.py, loaded INSIDE the rov_gui.control package so its relative imports (.pid,
    .rl_pwm, .attitude_ref, .geometry) are today's modules — the comparison isolates rl_policy.py. Its REPO_ROOT is the
    backup folder, so every policy_dir handed to it is absolute. None when the backup is not here."""
    if not BACKUP.is_file():
        return None
    spec = importlib.util.spec_from_file_location("rov_gui.control._rl_policy_pre_goalframe_20261015", str(BACKUP))
    mod = importlib.util.module_from_spec(spec)
    assert mod.__package__ == "rov_gui.control"
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# drivers shared with the fixture generator (RL_controller logs/eval/station_fix_20261015/impl-goalframe/scripts)
# =============================================================================
def builder_frames(ob, spec: dict, cases: dict):
    """RL_controller's ObsBuilder on the parity cases -> (frames (N, frame_dim), nu (N, 6)). Each case: a fresh builder
    state, the goal through set_goal (NED, tilted, as the station holds it), a fix captured NOW (age 0: its estimator
    propagates nothing) at the case pose, its LPF world velocity set to the case's ``v_w``. ``nu`` = the builder's own
    body velocity back in FRD + the rates, i.e. exactly the floats the station's state hands HwRl."""
    b = ob.ObsBuilder(spec)
    frames, nus = [], []
    for k in range(len(cases["yaw_ref"])):
        eta, rates = cases["eta"][k], cases["rates"][k]
        b.reset()
        b.set_goal(cases["p_ref"][k], float(cases["yaw_ref"][k]), v_goal_ned=cases["v_ref"][k],
                   yaw_rate_ned=float(cases["r_ref"][k]), roll_ned=float(cases["rp"][k][0]),
                   pitch_ned=float(cases["rp"][k][1]))
        b.fix_p, b.fix_yaw, b.fix_t = ob.C_NED_TO_UP @ np.asarray(eta[:3], float), -float(eta[5]), 0.0
        b.v_w = np.array(cases["v_w"][k], float)
        b.step(0.0, None, {"roll": float(eta[3]), "pitch": float(eta[4]), "p": float(rates[0]), "q": float(rates[1]),
                           "r": float(rates[2])}, prev_action=cases["prev"][k])
        v_b = b.last["v_b"]
        nus.append(np.array([v_b[0], -v_b[1], -v_b[2], rates[0], rates[1], rates[2]], float))
        frames.append(np.array(b.hist[0], float))
    return np.array(frames), np.array(nus)


def builder_sequence(ob, spec: dict, seq: dict):
    """One ObsBuilder across ``seq`` ticks (its history tiles on the first tick, newest first) -> (obs (T, obs_dim) in
    float64, nu (T, 6)); per tick as builder_frames, the fix captured at the tick's own time."""
    b = ob.ObsBuilder(spec)
    obs, nus = [], []
    for k in range(len(seq["t"])):
        eta, rates, t = seq["eta"][k], seq["rates"][k], float(seq["t"][k])
        b.set_goal(seq["p_ref"][k], float(seq["yaw_ref"][k]), v_goal_ned=seq["v_ref"][k],
                   yaw_rate_ned=float(seq["r_ref"][k]), roll_ned=float(seq["rp"][k][0]), pitch_ned=float(seq["rp"][k][1]))
        b.fix_p, b.fix_yaw, b.fix_t = ob.C_NED_TO_UP @ np.asarray(eta[:3], float), -float(eta[5]), t
        b.v_w = np.array(seq["v_w"][k], float)
        b.step(t, None, {"roll": float(eta[3]), "pitch": float(eta[4]), "p": float(rates[0]), "q": float(rates[1]),
                         "r": float(rates[2])}, prev_action=seq["prev"][k])
        v_b = b.last["v_b"]
        nus.append(np.array([v_b[0], -v_b[1], -v_b[2], rates[0], rates[1], rates[2]], float))
        obs.append(np.array(b.hist, float).reshape(-1))
    return np.array(obs), np.array(nus)


def station_frames(ctrl, cases: dict) -> np.ndarray:
    """HwRlPwm7d._frame on the parity cases: the flown roll/pitch reference and the fed-back action set as the builder's."""
    out = []
    for k in range(len(cases["yaw_ref"])):
        ctrl._rp_flown = (float(cases["rp"][k][0]), float(cases["rp"][k][1]))
        ctrl._prev_action = np.array(cases["prev"][k], float)
        out.append(ctrl._frame(np.asarray(cases["eta"][k], float), np.asarray(cases["nu"][k], float),
                               np.asarray(cases["p_ref"][k], float), float(cases["yaw_ref"][k]),
                               np.asarray(cases["v_ref"][k], float), float(cases["r_ref"][k])))
    return np.array(out)


def station_sequence(ctrl, seq: dict) -> np.ndarray:
    """HwRlPwm7d.observation over ``seq`` (reset first; the reference through set_target_ned, |r_ref| below the clip)."""
    ctrl.reset()
    out = []
    for k in range(len(seq["t"])):
        ctrl.set_target_ned(np.asarray(seq["p_ref"][k], float), float(seq["yaw_ref"][k]),
                            v_ned=np.asarray(seq["v_ref"][k], float), r_ned=float(seq["r_ref"][k]))
        ctrl._rp_flown = (float(seq["rp"][k][0]), float(seq["rp"][k][1]))
        ctrl._prev_action = np.array(seq["prev"][k], float)
        out.append(np.array(ctrl.observation(np.asarray(seq["eta"][k], float), np.asarray(seq["nu"][k], float),
                                             float(seq["t"][k])), float))
    return np.array(out)


def drive_legacy(ctrl, inp: dict, plan: bool = False) -> dict:
    """The whole tick, as the worker runs it: reference, ``step``, the applied command fed back (pseudo-wrench for the
    axes LOWs, the pulses for the pwm LOWs). Records what the NETWORK was fed (policy.act wrapped), the returned u and
    the pulses. ``plan`` = the reference as a one-stage NedPlan carrying ``rp`` (the 7d LOWs fly its roll/pitch)."""
    seen = []
    act = ctrl.policy.act

    def rec(obs):
        seen.append(np.array(obs, dtype=np.float64))
        return act(obs)

    ctrl.policy.act = rec
    ctrl.reset()
    us, pws = [], []
    for k in range(len(inp["t"])):
        p, yaw, v, r = (np.asarray(inp["p_ref"][k], float), float(inp["yaw_ref"][k]), np.asarray(inp["v_ref"][k], float),
                        float(inp["r_ref"][k]))
        if plan:
            ctrl.set_path_plan_ned(NedPlan(p_ned=p.reshape(3, 1), yaw_ned=np.array([yaw]), v_ned=v.reshape(3, 1),
                                           r_ned=np.array([r]), psi_path=np.zeros(1),
                                           rp_ned=np.asarray(inp["rp"][k], float).reshape(2, 1)))
        else:
            ctrl.set_target_ned(p, yaw, v_ned=v, r_ned=r)
        u, info = ctrl.step(np.asarray(inp["eta"][k], float), np.asarray(inp["nu"][k], float), np.zeros(6),
                            float(inp["t"][k]))
        us.append(np.asarray(u, dtype=np.float64))
        if "pwm_us" in info:
            pws.append(np.asarray(info["pwm_us"], dtype=np.float64))
            ctrl.note_applied_pwm(info["pwm_us"])
        else:
            ctrl.note_applied(u)
    ctrl.policy.act = act
    return {"obs": np.array(seen), "u": np.array(us),
            "pwm": np.array(pws, dtype=np.float64) if pws else np.zeros((0, 8))}


def _npz(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    return d, json.loads(str(d.pop("meta")))


def _sub(d: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


# =============================================================================
# (i) the layout gate
# =============================================================================
def test_frame_layout_is_rl_controllers_and_every_installed_export_passes_the_gate():
    """rl_policy.frame_layout spells RL_controller's rov_rl/obs_spec.frame_layout for every combination (live, by path,
    when the checkout is here), and every installed export / exporter fixture passes the gate in the frame its own
    obs_goal_frame names (the six exports installed before 2026-10-15 are world-frame ones)."""
    if RLC_OBS_SPEC.is_file():
        rlc = _load_by_path(RLC_OBS_SPEC, "rlc_obs_spec")
        for gv in (False, True):
            for n in (4, 6, 8):
                for gf in (False, True):
                    assert frame_layout(gv, n, goal_frame=gf) == rlc.frame_layout(gv, n, goal_frame=gf), (gv, n, gf)
    assert frame_layout(True, 8, goal_frame=True)[0] == Q_SLOTS_GOAL[0] == "q_des_goalframe(w,x,y,z)"
    assert frame_layout(True, 8, goal_frame=True)[2] == Q_SLOTS_GOAL[1] == "q_est_goalframe(w,x,y,z)"
    assert frame_layout(False, 4) == list(_spec(FIX / "rl7d_smoke_obs_spec.json")["frame_layout"][:5]) + ["prev_action(4)"]
    specs = sorted(POLICIES.glob("*/obs_spec.json")) + sorted(FIX.glob("rl*_obs_spec.json"))
    assert len(specs) >= 8, specs
    for p in specs:
        s = _spec(p)
        assert check_frame_layout(s, "test", p.parent.name) is bool(s.get("obs_goal_frame", False)), p
    assert check_frame_layout(_spec(ATT7_SPEC), "test", "att7") is True
    assert all(check_frame_layout(_spec(POLICIES / n / "obs_spec.json"), "test", n) is False
               for n in ("traj5_s2", "traj10_s1", "traj12_s2", "pwm10_s1", "pwm11k0.2_s0", "att7pwm_s0"))


@pytest.mark.parametrize("low", LOWS)
def test_layout_gate_accepts_each_layout_only_with_its_flag(low, tmp_path):
    world = _spec(WORLD_BASE[low])
    assert "obs_goal_frame" not in world and world["frame_layout"][0] == Q_SLOTS_WORLD[0]
    goal = _goal_twin(world)
    # accepted: the world layout with the flag absent or false, the goal layout with the flag true
    for i, (base, over, gf) in enumerate([(world, {}, False), (world, {"obs_goal_frame": False}, False),
                                          (goal, {}, True)]):
        said = []
        o = CLS[low](_cfg(low, _export(tmp_path, base, f"ok{i}", **over)), log=said.append)
        assert o.goal_frame is gf and o.meta()["obs_goal_frame"] is gf
        assert ("goal-frame quaternions" in said[-1]) is gf, said
        assert ("obs_goal_frame" in o.meta()["frames"]) is gf
        json.dumps(o.meta())
    mixed = [goal["frame_layout"][0]] + world["frame_layout"][1:]           # q_des in the goal frame, q_est not
    bad = [
        (goal, {"obs_goal_frame": _DROP}, r"GOAL-FRAME quaternion slots .* obs_goal_frame is absent"),
        (goal, {"obs_goal_frame": False}, r"GOAL-FRAME quaternion slots .* obs_goal_frame is false"),
        (world, {"obs_goal_frame": True}, r"says obs_goal_frame: true but its frame_layout names the WORLD-frame"),
        (goal, {"obs_goal_frame": "true"}, r"obs_goal_frame 'true' — expected true or false"),
        (goal, {"obs_goal_frame": 1}, r"obs_goal_frame 1 — expected true or false"),
        (goal, {"frame_layout": mixed}, r"frame_layout \[.*\] != \["),
        (world, {"frame_layout": mixed}, r"frame_layout \[.*\] != \["),
        (world, {"frame_layout": _DROP}, r"frame_layout None != \["),
        (world, {"frame_layout": world["frame_layout"][:-1]}, r"frame_layout \[.*\] != \["),
        (goal, {"frame_dim": goal["frame_dim"] + 1}, rf"frame_dim {goal['frame_dim'] + 1} != {goal['frame_dim']}"),
    ]
    for i, (base, over, why) in enumerate(bad):
        with pytest.raises(ValueError, match=why) as e:
            _build(low, _export(tmp_path, base, f"bad{i}", **over))
        assert str(e.value).startswith(f"{low}: policy "), str(e.value)
        assert "refused" in str(e.value), str(e.value)


@pytest.mark.parametrize("low", LOWS)
def test_only_the_two_quaternion_slots_differ_between_the_twins(low, tmp_path):
    """Same export, same state, same reference: the goal-frame twin's frame differs from the world-frame twin's ONLY in
    q_des / q_est, which are the world ones turned by -goal yaw (checked with rotation matrices, not the module's
    quaternion code), canonicalised; q_des carries no yaw, q_est the heading error. offset_b and goal_v_b stay in the
    body frame of the WORLD q_est (as trained)."""
    world = _spec(WORLD_BASE[low])
    ow = _build(low, _export(tmp_path, world, "w"))
    og = _build(low, _export(tmp_path, _goal_twin(world), "g"))
    rng = np.random.default_rng(7)
    for k in range(40):
        yaw_ref = float(rng.uniform(-math.pi, math.pi)) if k > 2 else [math.pi, -math.pi, math.radians(179.9)][k]
        eta = np.r_[rng.uniform(-1, 1, 3), rng.uniform(-0.3, 0.3, 2), float(_wrap(yaw_ref + rng.uniform(-0.6, 0.6)))]
        nu = rng.uniform(-0.3, 0.3, 6)
        p_ref, v_ref, r_ref = eta[:3] + rng.uniform(-0.4, 0.4, 3), rng.uniform(-0.1, 0.1, 3), float(rng.uniform(-0.3, 0.3))
        rp = (float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.2, 0.2))) if low.endswith("7d") else (0.0, 0.0)
        prev = rng.uniform(-1, 1, ow.n_act)
        f = {}
        for name, o in (("w", ow), ("g", og)):
            if low.endswith("7d"):
                o._rp_flown = rp
            o._prev_action = prev.copy()
            f[name] = o._frame(eta, nu, p_ref, yaw_ref, v_ref, r_ref)
        assert np.array_equal(f["g"][OFF], f["w"][OFF]) and np.array_equal(f["g"][REST], f["w"][REST])
        q_des = quat_from_euler_xyz(rp[0], -rp[1], -yaw_ref)
        q_est = quat_from_euler_xyz(eta[3], -eta[4], -eta[5])
        assert np.allclose(f["w"][QD], _unique(q_des), atol=1e-12) and np.allclose(f["w"][QE], _unique(q_est), atol=1e-12)
        yg = math.atan2(_R(q_des)[1, 0], _R(q_des)[0, 0])
        assert abs(_wrap(yg - (-yaw_ref))) < 1e-9                       # the goal yaw is the reference yaw (z-up)
        assert np.allclose(_R(f["g"][QD]), _Rz(-yg) @ _R(q_des), atol=1e-12)
        assert np.allclose(_R(f["g"][QE]), _Rz(-yg) @ _R(q_est), atol=1e-12)
        assert f["g"][QD][0] >= 0.0 and f["g"][QE][0] >= 0.0
        Rg = _R(f["g"][QD])
        assert abs(math.atan2(Rg[1, 0], Rg[0, 0])) < 1e-12               # no yaw left in the goal
        if not low.endswith("7d"):
            assert np.allclose(f["g"][QD], [1.0, 0.0, 0.0, 0.0], atol=1e-12)   # a level goal is the identity
        Re = _R(f["g"][QE])
        err = _wrap(-eta[5] - (-yaw_ref))                                 # heading error, z-up (CCW +)
        assert abs(_wrap(math.atan2(Re[1, 0], Re[0, 0]) - err)) < 1e-9


def test_the_att7_export_flies_goal_frame_on_both_pwm_lows(tmp_path):
    """The fixture is the att7 export's obs_spec.json verbatim (checked live when the export is here); it builds on rl_pwm
    and rl_pwm_7d with goal-frame quaternions, says so in the log and meta, and is refused on the axes LOWs for its
    action mode (the gate does not change who flies what)."""
    spec = _spec(ATT7_SPEC)
    exported = sorted(RLC.glob("logs/rsl_rl/rov_posehold/*_pwm11att7_s0/exported"))
    if exported:
        assert spec == _spec(exported[-1] / "obs_spec.json"), "the fixture is the export's obs_spec.json, verbatim"
    assert spec["obs_goal_frame"] is True and spec["frame_layout"] == frame_layout(True, 8, goal_frame=True)
    assert spec["attitude_reference"]["trained"] is True and spec["action_mode"] == "pwm"
    pdir = _export(tmp_path, spec, "att7")
    for low in ("rl_pwm", "rl_pwm_7d"):
        said = []
        o = CLS[low](_cfg(low, pdir), log=said.append)
        assert o.goal_frame is True and "goal-frame quaternions" in said[-1], said
        m = o.meta()
        json.dumps(m)
        assert m["obs_goal_frame"] is True and m["frame_layout"][0] == "q_des_goalframe(w,x,y,z)"
    for low in ("rl", "rl_7d"):
        with pytest.raises(ValueError, match="'pwm' policy"):
            _build(low, pdir)
    if exported:                                       # the real export (its own weights) builds where it will fly
        o = _build("rl_pwm_7d", exported[-1])
        assert o.goal_frame is True and o.policy.obs_dim == 116 and o.pwm_cap == pytest.approx(0.5)


# =============================================================================
# (ii) parity with RL_controller deploy/obs_builder.py (att7 export)
# =============================================================================
def _att7_low(tmp):
    """HwRlPwm7d on the att7 export (its verbatim obs_spec.json, zero weights). leash_m 100: the station's goal leash (a
    station-only clamp, pinned in test_control.py) never acts, so offsets beyond 1 m meet the trained clip exactly as
    in the builder."""
    return _build("rl_pwm_7d", _export(tmp, _spec(ATT7_SPEC), "att7"), leash_m=100.0)


def test_parity_fixture_is_the_att7_builders_output():
    d, meta = _npz(PARITY)
    assert meta["obs_spec_sha256"] == _sha256(ATT7_SPEC), "the fixture was made for this obs_spec.json"
    heads = np.degrees(d["yaw_ref"])
    for h in (179.9, 180.0, -179.9):
        assert np.sum(np.isclose(heads, h, atol=1e-9)) >= 4, h        # the named headings are in it, several times each
    tilted = np.abs(d["rp"]).max(axis=1) > 0.05
    assert tilted.sum() >= 50 and (~tilted).sum() >= 10
    assert np.isclose(np.abs(d["rp"][:, 0]).max(), 0.14) and np.isclose(np.abs(d["rp"][:, 1]).max(), 0.26)  # corners
    ob = _rlc_builder()
    if ob is not None:                                 # live: the builder still produces exactly these numbers
        spec = _spec(ATT7_SPEC)
        frames, nu = builder_frames(ob, spec, d)
        assert np.array_equal(frames, d["frame"]) and np.array_equal(nu, d["nu"])
        obs, nu_s = builder_sequence(ob, spec, _sub(d, "seq_"))
        assert np.array_equal(obs, d["seq_obs"]) and np.array_equal(nu_s, d["seq_nu"])


def test_att7_frames_match_rl_controllers_builder(tmp_path):
    """Every frame of the parity set — random states, the named headings, tilted and level goals, offsets past the
    1 m clip — is the builder's, float for float."""
    d, _meta = _npz(PARITY)
    got = station_frames(_att7_low(tmp_path), d)
    assert got.shape == d["frame"].shape == (len(d["yaw_ref"]), 29)
    bad = np.flatnonzero(~np.all(got == d["frame"], axis=1))
    assert bad.size == 0, (bad[:5], np.abs(got - d["frame"]).max())
    assert np.all(got[:, QD.start] >= 0.0) and np.all(got[:, QE.start] >= 0.0)
    ob = _rlc_builder()
    if ob is not None:
        frames, nu = builder_frames(ob, _spec(ATT7_SPEC), d)
        assert np.array_equal(station_frames(_att7_low(tmp_path / "live"), {**d, "nu": nu}), frames)


def test_att7_history_matches_rl_controllers_builder(tmp_path):
    """The stacked observation over a run whose goal heading crosses +-180 deg (tilted goals): newest first, tiled on
    the first tick — the builder's, float for float."""
    d, _meta = _npz(PARITY)
    seq = _sub(d, "seq_")
    assert np.degrees(seq["yaw_ref"]).max() > 179.0 and np.degrees(seq["yaw_ref"]).min() < -179.0
    got = station_sequence(_att7_low(tmp_path), seq)
    assert got.shape == seq["obs"].shape and np.array_equal(got, seq["obs"]), np.abs(got - seq["obs"]).max()
    ob = _rlc_builder()
    if ob is not None:
        obs, nu = builder_sequence(ob, _spec(ATT7_SPEC), seq)
        assert np.array_equal(station_sequence(_att7_low(tmp_path / "live"), {**seq, "nu": nu}), obs)


def test_goal_frame_quats_is_the_builders_transform():
    """The helper against the builder's own lines (live) and against the definition (rotation matrices) on random
    tilted goals, including goal yaws at exactly +-pi."""
    rng = np.random.default_rng(3)
    ob = _rlc_builder()
    for k in range(500):
        yaw = [math.pi, -math.pi, math.radians(179.9), math.radians(-179.9)][k] if k < 4 else rng.uniform(-math.pi, math.pi)
        q_des = quat_from_euler_xyz(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), yaw)
        q_est = quat_from_euler_xyz(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-math.pi, math.pi))
        qd, qe = goal_frame_quats(q_des, q_est)
        yg = yaw_from_q(q_des)
        assert np.allclose(_R(qd), _Rz(-yg) @ _R(q_des), atol=1e-12) and np.allclose(_R(qe), _Rz(-yg) @ _R(q_est), atol=1e-12)
        if ob is not None:
            assert yg == ob.yaw_from_q(q_des)
            q_rot = np.array([math.cos(-0.5 * yg), 0.0, 0.0, math.sin(-0.5 * yg)])
            assert np.array_equal(qd, ob.quat_mul_np(q_rot, q_des)) and np.array_equal(qe, ob.quat_mul_np(q_rot, q_est))


# =============================================================================
# (iii) continuity across +-180 deg
# =============================================================================
def _sweep(o, goal_yaws, veh_yaws, rp=(0.08, -0.15)):
    """Frames along a sweep (state and reference otherwise fixed; the 7d LOWs fly a tilted goal)."""
    tilted = isinstance(o, (HwRl7d, HwRlPwm7d))
    o._prev_action = np.linspace(-0.5, 0.5, o.n_act)
    out = []
    for yg, yv in zip(goal_yaws, veh_yaws):
        if tilted:
            o._rp_flown = rp
        out.append(o._frame(np.array([0.2, -0.1, 0.8, 0.05, -0.07, float(yv)]),
                            np.array([0.05, -0.02, 0.01, 0.02, -0.03, 0.04]), np.array([0.5, 0.2, 0.9]), float(yg),
                            np.array([0.03, -0.02, 0.0]), 0.1))
    return np.array(out)


@pytest.mark.parametrize("low", LOWS)
def test_goal_frame_observation_is_continuous_across_180(low, tmp_path):
    world = _spec(WORLD_BASE[low])
    og = _build(low, _export(tmp_path, _goal_twin(world), "g"))
    ow = _build(low, _export(tmp_path, world, "w"))
    goal = _wrap(np.radians(np.linspace(176.0, 184.0, 161)))           # 0.05 deg steps through +-180
    # (a) the goal heading crosses +-180 with the vehicle 3 deg off it: the quaternion slots do not move at all
    f = _sweep(og, goal, _wrap(goal + math.radians(3.0)))
    dq = np.abs(np.diff(f[:, np.r_[0:4, 7:11]], axis=0)).max()
    assert dq < 1e-9, dq
    assert np.abs(np.diff(f, axis=0)).max() < 2e-3                     # the rest turns with the hull, 0.05 deg a step
    fw = _sweep(ow, goal, _wrap(goal + math.radians(3.0)))
    assert np.abs(np.diff(fw[:, np.r_[0:4, 7:11]], axis=0)).max() > 1.0  # the world frame flips there (why att7 exists)
    # (b) the vehicle heading crosses +-180 under a goal at 180 deg: q_est carries the -4..+4 deg error smoothly
    veh = _wrap(np.radians(np.linspace(176.0, 184.0, 161)))
    f = _sweep(og, np.full(161, math.pi), veh)
    assert np.abs(np.diff(f, axis=0)).max() < 1e-3
    assert np.abs(np.diff(f[:, 0:4], axis=0)).max() < 1e-12            # the goal itself is constant
    fw = _sweep(ow, np.full(161, math.pi), veh)
    assert np.abs(np.diff(fw[:, 7:11], axis=0)).max() > 1.0
    # (c) +180 and -180 are one goal; 179.9 / 180 / -179.9 are three neighbouring goals
    f = _sweep(og, np.radians([180.0, -180.0, 179.9, -179.9]), np.full(4, math.radians(179.95)))
    assert np.abs(f[0] - f[1]).max() < 1e-12
    assert np.abs(f[2] - f[0]).max() < 2e-3 and np.abs(f[3] - f[0]).max() < 2e-3


# =============================================================================
# (iv) world-frame exports: byte-identical to the module before 2026-10-15
# =============================================================================
#: (LOW, export dir, plan?) flown against the pre-change module when it is here; the first two are also pinned in the
#: legacy fixture.
LEGACY_EXPORTS = [("rl", POLICIES / "traj12_s2", False), ("rl_pwm", POLICIES / "pwm11k0.2_s0", False),
                  ("rl", POLICIES / "traj10_s1", False), ("rl", POLICIES / "traj5_s2", False),
                  ("rl_pwm", POLICIES / "pwm10_s1", False), ("rl_pwm_7d", POLICIES / "att7pwm_s0", True)]


def test_legacy_fixture_world_frame_exports_are_byte_identical():
    d, meta = _npz(LEGACY)
    for name, low in (("traj12_s2", "rl"), ("pwm11k0.2_s0", "rl_pwm")):
        pdir = POLICIES / name
        assert meta["exports"][name]["obs_spec_sha256"] == _sha256(pdir / "obs_spec.json"), f"{name} changed: regenerate"
        assert meta["exports"][name]["weights_sha256"] == _sha256(pdir / "policy_weights.npz"), f"{name} changed"
        o = _build(low, pdir)
        assert o.goal_frame is False
        inp = _sub(d, f"{name}__")
        got = drive_legacy(o, inp)
        for k in ("obs", "u", "pwm"):
            assert got[k].shape == inp[k].shape, (name, k, got[k].shape, inp[k].shape)
            assert got[k].tobytes() == inp[k].tobytes(), (name, k, np.abs(got[k] - inp[k]).max())
        assert np.degrees(np.abs(inp["yaw_ref"])).max() > 179.0         # the world-frame flip is inside the run


def _legacy_inputs(n: int, seed: int) -> dict:
    """A run for the live comparison: headings sweeping through +-180 deg, goals inside and beyond the leash, every
    4th tick r_ref == 0 (the station's yaw-rate recovery), tilted plan references for the 7d LOWs."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    yaw_ref = _wrap(np.radians(172.0 + 0.8 * k))
    eta = np.c_[0.3 + 0.02 * np.cumsum(rng.normal(size=(n, 3)), axis=0), rng.uniform(-0.1, 0.1, (n, 2)),
                _wrap(np.radians(170.0 + 1.0 * k))]
    r_ref = rng.uniform(-0.3, 0.3, n)
    r_ref[::4] = 0.0
    return {"t": 0.05 * k, "eta": eta, "nu": rng.uniform(-0.2, 0.2, (n, 6)),
            "p_ref": eta[:, :3] + rng.uniform(-1.2, 1.2, (n, 3)), "yaw_ref": yaw_ref,
            "v_ref": rng.uniform(-0.1, 0.1, (n, 3)), "r_ref": r_ref,
            "rp": np.c_[rng.uniform(-0.15, 0.15, n), rng.uniform(-0.3, 0.3, n)]}


@pytest.mark.parametrize("low,pdir,plan", LEGACY_EXPORTS + [("rl_7d", None, True)],
                         ids=[f"{lo}-{p.name}" for lo, p, _ in LEGACY_EXPORTS] + ["rl_7d-smoke"])
def test_world_frame_exports_are_byte_identical_to_the_pre_change_module(low, pdir, plan, tmp_path):
    old = _backup_module()
    if old is None:
        pytest.skip(f"pre-change copy not on this machine ({BACKUP}); the legacy fixture test still pins two exports")
    if pdir is None:                                   # rl_7d: the exporter's smoke spec, zero weights (no rl_7d export)
        pdir = _export(tmp_path, _spec(FIX / "rl7d_smoke_obs_spec.json"), "rl7d")
    inp = _legacy_inputs(48, seed=11)
    new_c = _build(low, pdir)
    old_c = _build(low, pdir, cls=getattr(old, CLS[low].__name__))
    assert new_c.goal_frame is False and not hasattr(old_c, "goal_frame")
    a, b = drive_legacy(new_c, inp, plan=plan), drive_legacy(old_c, inp, plan=plan)
    for k in ("obs", "u", "pwm"):
        assert a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes(), (low, pdir, k)
    assert len(a["obs"]) == 48
    # and observation() alone, fresh history, the same calls
    for c in (new_c, old_c):
        c.reset()
    for k in range(12):
        for c in (new_c, old_c):
            c.set_target_ned(inp["p_ref"][k], float(inp["yaw_ref"][k]), v_ned=inp["v_ref"][k], r_ned=float(inp["r_ref"][k]))
        assert new_c.observation(inp["eta"][k], inp["nu"][k], float(inp["t"][k])).tobytes() == \
            old_c.observation(inp["eta"][k], inp["nu"][k], float(inp["t"][k])).tobytes()
    # meta: one key more, nothing else changed (policy_dir is relative to each module's REPO_ROOT: compared absolute)
    mn, mo = new_c.meta(), old_c.meta()
    assert new_c.policy_dir == old_c.policy_dir and mn.pop("policy_dir") and mo.pop("policy_dir")
    # (2026-10-16, R7: what velocity / observation guard the export says it was trained on — none of these exports
    # carries either block, so both read "none")
    assert mn.pop("velocity_source") == {"export": None, "station": None}
    assert mn.pop("obs_guard_trained", None) is None
    assert mn.pop("obs_goal_frame") is False and mn == mo
