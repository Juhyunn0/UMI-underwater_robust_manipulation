#!/usr/bin/env python3
"""
test_dp_policy.py — the diffusion-policy session (perception/dp_policy.py).

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_dp_policy.py
    ~/miniforge3/envs/robust/bin/python      rov_gui/tests/test_dp_policy.py   (no torch: stub half only)

Two halves. The first needs nothing (numpy): the module imports without
torch, the safe ``eval`` resolver is arithmetic-only, the re-implemented
normalizer round-trips, the STUB session honours the same contract the real
one does (the 5-dim ``pos_yaw_width`` action by default, the legacy
``pose10d`` and the 7-dim ``pos_rpy_width`` (6-DoF variant, 2026-09-26)
only by ctor kwarg, a ``dyaw_rate_rad_s`` yaw ramp and a roll/pitch ramp on
the 7-dim stub), the action-representation vocabulary is the ONE in
``rov_gui.state`` (shared with control/policy_frames; {10, 5, 7} since
2026-09-26 -- a bare width 7 is now a LEGAL inference, which the old
version of this file asserted was an error), the loader's
``resolve_action_repr`` rules hold, the identity-normaliser refusal applies
to every non-pose10d representation, the ``rpy_convention`` contract key is
read, the factory picks the stub for ``ckpt: stub`` / ``stub7`` /
``stub_rp``, and the upstream import leaves the bare ``Utils`` name
(FoundationStereo / FoundationPose) exactly as it was. The second half loads the REAL checkpoint on the GPU and skips —
saying why — when there is no CUDA, no checkpoint or no UMI checkout:
strict load, the obs/action contract values, output shape and dtype, two
seeded predicts identical (dropout and BatchNorm are off on the calling
thread), inference under 500 ms, and the ``describe()`` record the run meta
will carry. Since 2026-09-07 the contract asserted there is the 5-dim one
(action_dim 5, ``pos_yaw_width``, every output column inside the
normalizer's input_stats, no identity normalizer column, the C3 mount and
the epoch in the contract): on the legacy 10-dim checkpoint those tests
FAIL rather than skip, on purpose — a stale ``POLICY_CKPT_DEFAULT`` is what
they exist to catch. Cross-interpreter parity with the training environment
is the separate tool ``rov_gui/tools/dp_policy_parity.py``.
"""

from __future__ import annotations

import atexit
import importlib
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from rov_gui import state as S
from rov_gui.perception import dp_policy
from rov_gui.perception.dp_policy import (DpPolicySession, StubPolicySession,
                                          _Normalizer, identity_normaliser_columns,
                                          is_stub_ckpt, make_policy_session,
                                          resolve_action_repr, rpy_convention_of,
                                          safe_eval)


def _skip(reason: str):
    raise unittest.SkipTest(reason)


# ------------------------------------------------------------- module level
def test_module_imports_without_torch():
    """``rov_gui.perception`` stays free of torch at import; the session
    imports it inside load(). Checked in a fresh interpreter so an earlier
    test's import cannot mask it."""
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        import rov_gui.perception.dp_policy as m
        assert "torch" not in sys.modules, "torch imported at module scope"
        assert "diffusers" not in sys.modules
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-800:]


def test_safe_eval_accepts_arithmetic_and_rejects_names_and_calls():
    assert safe_eval("2 * 3 / 4") == 1.5
    assert safe_eval("-(1 + 2) * 3") == -9
    assert safe_eval("16 * 2 - 1") == 31
    assert safe_eval(7) == 7                       # non-str goes through str()
    for bad in ("__import__('os')", "x + 1", "max(1, 2)", "2 ** 8",
                "'a' * 3", "(lambda: 1)()", "1 if 1 else 0", "[1, 2]"):
        try:
            safe_eval(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"safe_eval accepted {bad!r}")
    # the copy in the reference tool must agree (same function, two files)
    from rov_gui.tools.dp_policy_reference import safe_eval as ref_eval
    for expr in ("2*3/4", "-(1+2)*3", "16 * 2 - 1"):
        assert ref_eval(expr) == safe_eval(expr)


def test_normalizer_round_trip_on_random_params():
    """normalize = x*scale+offset on reshape(-1, D); unnormalize inverts it
    for any leading shape; params grouped from state-dict keys; image key
    with scale 1 / offset 0 is the identity through the same formula."""
    rng = np.random.default_rng(3)
    sd = {}
    dims = {"robot0_eef_pos": 3, "robot0_gripper_width": 1, "action": 10,
            "camera0_depth": 1}
    for k, D in dims.items():
        sd[f"normalizer.params_dict.{k}.scale"] = (
            np.ones(D) if k == "camera0_depth"
            else rng.uniform(0.5, 60.0, D)).astype(np.float32)
        sd[f"normalizer.params_dict.{k}.offset"] = (
            np.zeros(D) if k == "camera0_depth"
            else rng.normal(0.0, 2.0, D)).astype(np.float32)
        sd[f"normalizer.params_dict.{k}.input_stats.min"] = rng.normal(size=D)
        sd[f"normalizer.params_dict.{k}.input_stats.max"] = rng.normal(size=D)
    sd["obs_encoder.some.weight"] = np.zeros(2)              # ignored
    n = _Normalizer.from_state_dict(sd)
    assert set(n.keys) == set(dims)
    for k, D in dims.items():
        assert n.dim(k) == D
        x = rng.normal(size=(2, 5, D)).astype(np.float32)
        y = n.normalize(k, x)
        assert y.shape == x.shape
        s, o = n.params(k)
        assert np.allclose(y, x * s + o, atol=1e-6)
        assert np.allclose(n.unnormalize(k, y), x, atol=1e-5)
        assert set(n.input_stats(k)) == {"min", "max"}
    img = rng.random((1, 2, 3, 8, 8)).astype(np.float32)
    assert np.array_equal(n.normalize("camera0_depth", img), img)
    d = n.normalize_dict({"action": np.zeros((16, 10), np.float32)})
    assert np.allclose(d["action"], n.params("action")[1])
    try:
        _Normalizer({"k": {"scale": np.ones(3), "offset": np.ones(2)}})
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched scale/offset accepted")


# ------------------------------------------------------------- the stub
def _stub_obs(c):
    obs = {c["image_key"]: np.zeros((c["obs_horizon"],) + tuple(c["image_shape"]),
                                    np.float32)}
    for k, shp in c["lowdim_shapes"].items():
        obs[k] = np.zeros((c["obs_horizon"],) + tuple(shp), np.float32)
    return obs


def test_stub_session_contract_and_predict():
    s = StubPolicySession("stub", dataset_fps=30.0)
    assert not s.ready
    logs = []
    s.start_async(logs.append)
    assert s.ready and not s.loading and s.error == "" and logs
    c = s.contract
    assert c["obs_keys"] == ["camera0_depth", "robot0_eef_pos",
                             "robot0_eef_rot_axis_angle", "robot0_gripper_width",
                             "robot0_eef_rot_axis_angle_wrt_start"]
    assert c["obs_horizon"] == 2 and c["action_horizon"] == 16
    # the FLOWN contract (2026-09-07): 5-dim [dx, dy, dz, dyaw, width]
    assert c["action_dim"] == 5 and c["down_sample_steps"] == 2
    assert c["action_repr"] == "pos_yaw_width" == S.POLICY_ACTION_REPR
    assert set(c["action_range"]) == {"min", "max"}
    assert len(c["action_range"]["min"]) == 5 == len(c["action_range"]["max"])
    assert c["yaw_axis_cam_tilt_deg"] is None and c["yaw_axis_R_frd_cam"] is None
    assert c["rpy_convention"] is None                # not a 7-dim contract
    assert c["ckpt_epoch"] is None and c["ckpt_global_step"] is None
    assert abs(c["obs_dt_s"] - 2.0 / 30.0) < 1e-12 and c["fps"] == 30.0
    assert c["image_shape"] == (3, 224, 224)
    assert c["lowdim_shapes"] == {"robot0_eef_pos": (3,),
                                  "robot0_eef_rot_axis_angle": (6,),
                                  "robot0_gripper_width": (1,),
                                  "robot0_eef_rot_axis_angle_wrt_start": (6,)}
    act, ms = s.predict(_stub_obs(c))
    assert act.shape == (16, 5) and act.dtype == np.float32
    assert np.isfinite(ms) and ms >= 0.0
    k = np.arange(16)
    assert np.allclose(act[:, 2], 0.05 * k * c["obs_dt_s"])     # TCP +z forward
    assert np.all(act[:, :2] == 0.0)
    assert np.all(act[:, 3] == 0.0)                              # dyaw 0: straight
    assert np.allclose(act[:, 4], 0.069)                         # OPEN width
    assert abs(act[-1, 2] - 0.05) < 1e-6                         # 1.0 s -> 5 cm
    lo, hi = (np.asarray(c["action_range"][n]) for n in ("min", "max"))
    assert np.all(act >= lo - 1e-9) and np.all(act <= hi + 1e-9)  # inside its own range
    d = s.describe()
    assert d["stub"] is True and d["n_predict"] == 1
    assert d["contract"] == c
    assert d["action_repr"] == "pos_yaw_width" and d["stub_dyaw_rate_rad_s"] == 0.0
    # the stub validates obs like the real session (a worker bug shows here)
    bad = _stub_obs(c)
    bad["robot0_eef_pos"] = np.zeros((2, 4), np.float32)
    try:
        s.predict(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("stub accepted a wrong lowdim shape")
    del bad["camera0_depth"]
    bad["robot0_eef_pos"] = np.zeros((2, 3), np.float32)
    try:
        s.predict(bad)
    except KeyError:
        pass
    else:
        raise AssertionError("stub accepted a missing image key")
    s.close()
    assert not s.ready


def test_stub_session_legacy_pose10d():
    """``action_repr="pose10d"`` is the ctor kwarg and nothing else: the old
    (16, 10) chunk with identity rot6d in 3:9 and the width in 9 (for the
    recomposition tools); a name outside the vocabulary raises."""
    s = StubPolicySession("stub", action_repr="pose10d")
    s.load()
    c = s.contract
    assert c["action_dim"] == 10 and c["action_repr"] == "pose10d"
    assert len(c["action_range"]["min"]) == 10
    act, _ = s.predict(_stub_obs(c))
    assert act.shape == (16, 10) and act.dtype == np.float32
    assert np.allclose(act[:, 3:9], [1, 0, 0, 0, 1, 0])          # identity rot6d
    assert np.allclose(act[:, 9], 0.069)
    assert np.allclose(act[:, 2], 0.05 * np.arange(16) * c["obs_dt_s"])
    assert s.describe()["action_repr"] == "pose10d"
    for bogus in ("pose6d", "", None, 5):
        try:
            StubPolicySession("stub", action_repr=bogus)
        except ValueError:
            pass
        else:
            raise AssertionError(f"stub accepted action_repr {bogus!r}")
    try:
        StubPolicySession("stub", dyaw_rate_rad_s=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("stub accepted a NaN dyaw rate")


def test_stub_dyaw_rate():
    """``dyaw_rate_rad_s`` ramps column 3 (dyaw) at that rate on the obs_dt
    grid -- the way the position column ramps -- so a test can push a known
    yaw through the composition seam."""
    s = StubPolicySession("stub", dyaw_rate_rad_s=0.1)
    s.load()
    c = s.contract
    act, _ = s.predict(_stub_obs(c))
    k = np.arange(16)
    assert act.shape == (16, 5)
    assert np.allclose(act[:, 3], 0.1 * k * c["obs_dt_s"])
    assert abs(act[-1, 3] - 0.1) < 1e-6                          # 1.0 s -> 0.1 rad
    assert np.allclose(act[:, 2], 0.05 * k * c["obs_dt_s"])      # unchanged
    assert np.allclose(act[:, 4], 0.069)
    assert s.describe()["stub_dyaw_rate_rad_s"] == 0.1
    s2 = StubPolicySession("stub", dyaw_rate_rad_s=-0.2)
    s2.load()
    a2, _ = s2.predict(_stub_obs(c))
    assert np.allclose(a2[:, 3], -0.2 * k * c["obs_dt_s"])
    # the legacy stub ignores the rate (no yaw column to ramp)
    s3 = StubPolicySession("stub", action_repr="pose10d", dyaw_rate_rad_s=0.3)
    s3.load()
    a3, _ = s3.predict(_stub_obs(s3.contract))
    assert np.allclose(a3[:, 3:9], [1, 0, 0, 0, 1, 0])


def test_action_repr_vocabulary_is_shared():
    """ONE vocabulary: dp_policy spells the representation through
    rov_gui.state, and control/policy_frames infers the same names from the
    width -- so the loader, the stub, the intake gate and compose_plan can
    never disagree on what a (16, 5) or a (16, 7) means. Since 2026-09-26
    the vocabulary is {10, 5, 7}; the flown default stays 5-dim."""
    from rov_gui.control.policy_frames import action_repr_of

    assert dp_policy.ACTION_REPR_BY_DIM is S.ACTION_REPR_BY_DIM
    assert dp_policy.ACTION_DIM_BY_REPR is S.ACTION_DIM_BY_REPR
    assert S.ACTION_REPR_BY_DIM == {10: "pose10d", 5: "pos_yaw_width",
                                    7: "pos_rpy_width"}
    assert action_repr_of(5) == "pos_yaw_width" == S.ACTION_REPR_POS_YAW_WIDTH
    assert action_repr_of(10) == "pose10d" == S.ACTION_REPR_POSE10D
    assert action_repr_of(7) == "pos_rpy_width" == S.ACTION_REPR_POS_RPY_WIDTH
    assert StubPolicySession("stub").contract["action_repr"] == S.POLICY_ACTION_REPR
    assert S.POLICY_ACTION_REPR == "pos_yaw_width"            # the flown default
    assert set(S.POLICY_ACTION_REPRS_FLYABLE) == {"pos_yaw_width", "pos_rpy_width"}
    assert set(dp_policy.STUB_ACTION_RANGE) == set(S.ACTION_DIM_BY_REPR)
    for r, (lo, hi) in dp_policy.STUB_ACTION_RANGE.items():
        assert len(lo) == len(hi) == S.ACTION_DIM_BY_REPR[r]
        assert np.all(np.asarray(hi) > np.asarray(lo))
    # the 7-dim stub range is the 5-dim one with two angle columns spliced in
    lo5, hi5 = dp_policy.STUB_ACTION_RANGE["pos_yaw_width"]
    lo7, hi7 = dp_policy.STUB_ACTION_RANGE["pos_rpy_width"]
    assert lo7[:4] + [lo7[6]] == lo5 and hi7[:4] + [hi7[6]] == hi5
    assert lo7[4:6] == [-math.pi / 2] * 2 and hi7[4:6] == [math.pi / 2] * 2
    # the ONE checkpoint path
    assert dp_policy.DEFAULT_CKPT == S.POLICY_CKPT_DEFAULT
    assert dp_policy.DEFAULT_UMI_REPO == S.POLICY_UMI_REPO


def test_resolve_action_repr_rules():
    """The loader's resolution rules: declared first, width second, never
    both disagreeing. REWRITTEN 2026-09-26: ``({}, 7)`` used to be asserted
    an error (7 was outside the vocabulary); it is now the legal inference
    ``pos_rpy_width`` -- the whole point of growing the vocabulary -- while a
    declared name that contradicts the width still raises."""
    assert resolve_action_repr({"shape": [5], "action_repr": "pos_yaw_width"}, 5) \
        == "pos_yaw_width"
    assert resolve_action_repr({"shape": [7], "action_repr": "pos_rpy_width"}, 7) \
        == "pos_rpy_width"
    assert resolve_action_repr({"shape": [10]}, 10) == "pose10d"
    assert resolve_action_repr({"shape": [5]}, 5) == "pos_yaw_width"
    assert resolve_action_repr({}, 7) == "pos_rpy_width"          # legal since 2026-09-26
    assert resolve_action_repr({"shape": [7]}, 7) == "pos_rpy_width"
    for meta, dim in (({"action_repr": "pose10d"}, 5),
                      ({"action_repr": "pos_yaw_width"}, 10),
                      ({"action_repr": "pose6d"}, 5),
                      ({"action_repr": "pose10d"}, 7),          # still wrong
                      ({"action_repr": "pos_yaw_width"}, 7),    # 5-dim name, 7 wide
                      ({"action_repr": "pos_rpy_width"}, 5),    # 7-dim name, 5 wide
                      ({}, 6), ({}, 8), ({}, 4)):               # widths outside {5,7,10}
        try:
            resolve_action_repr(meta, dim)
        except ValueError:
            pass
        else:
            raise AssertionError(f"resolve_action_repr accepted {meta}, {dim}")


def test_stub_session_pos_rpy_width_and_the_rp_ramp():
    """The 7-dim stub (6-DoF variant, 2026-09-26): columns 0:4 and 6 are the
    5-dim stub's columns 0:4 and 4 exactly, the roll/pitch columns 4:6 are
    identically 0 by default and a linear ramp to ``*_ramp_deg`` at the last
    knot when asked (``stub_rp``); the contract carries the C3 mount and the
    station's rpy_convention (the mount check REQUIRES both for this repr),
    and a ramp on a stub without roll/pitch columns is refused."""
    s5 = StubPolicySession("stub", dyaw_rate_rad_s=0.1)
    s7 = StubPolicySession("stub7", action_repr="pos_rpy_width", dyaw_rate_rad_s=0.1)
    s5.load(), s7.load()
    c = s7.contract
    assert c["action_dim"] == 7 and c["action_repr"] == "pos_rpy_width"
    assert len(c["action_range"]["min"]) == 7 == len(c["action_range"]["max"])
    assert c["rpy_convention"] == dp_policy.RPY_CONVENTION_STATION
    assert c["rpy_convention"] is not dp_policy.RPY_CONVENTION_STATION   # a copy
    from rov_gui.control.policy_frames import R_BT_C3, YAW_AXIS_CAM_TILT_DEG
    assert c["yaw_axis_cam_tilt_deg"] == YAW_AXIS_CAM_TILT_DEG
    assert np.allclose(np.asarray(c["yaw_axis_R_frd_cam"]), R_BT_C3, atol=0.0)
    a5, _ = s5.predict(_stub_obs(s5.contract))
    a7, _ = s7.predict(_stub_obs(c))
    assert a7.shape == (16, 7) and a7.dtype == np.float32
    assert np.array_equal(a7[:, :4], a5[:, :4]) and np.array_equal(a7[:, 6], a5[:, 4])
    assert np.all(a7[:, 4:6] == 0.0)                        # level: dropped-and-logged case
    lo, hi = (np.asarray(c["action_range"][n]) for n in ("min", "max"))
    assert np.all(a7 >= lo - 1e-9) and np.all(a7 <= hi + 1e-9)
    d = s7.describe()
    assert d["action_repr"] == "pos_rpy_width" and d["stub_rp_ramp_deg"] == (0.0, 0.0)
    # the ramp: 0 at knot 0, exactly *_ramp_deg at knot 15, linear between
    sr = StubPolicySession("stub_rp", action_repr="pos_rpy_width",
                           roll_ramp_deg=-2.0, pitch_ramp_deg=5.0)
    sr.load()
    ar, _ = sr.predict(_stub_obs(sr.contract))
    frac = np.arange(16) / 15.0
    assert np.allclose(ar[:, 4], math.radians(-2.0) * frac, atol=1e-7)
    assert np.allclose(ar[:, 5], math.radians(5.0) * frac, atol=1e-7)
    assert ar[0, 4] == 0.0 and ar[0, 5] == 0.0
    assert abs(math.degrees(ar[-1, 5]) - 5.0) < 1e-5
    s5b = StubPolicySession("stub")
    s5b.load()
    assert np.array_equal(ar[:, :4], s5b.predict(_stub_obs(sr.contract))[0][:, :4])  # cols 0:4 untouched
    assert sr.describe()["stub_rp_ramp_deg"] == (-2.0, 5.0)
    for bad in (dict(pitch_ramp_deg=5.0), dict(roll_ramp_deg=1.0),
                dict(action_repr="pose10d", pitch_ramp_deg=5.0)):
        try:
            StubPolicySession("stub", **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"stub accepted a roll/pitch ramp without the columns: {bad}")
    try:
        StubPolicySession("stub", action_repr="pos_rpy_width", pitch_ramp_deg=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("stub accepted a NaN pitch ramp")
    # the 5-dim and pose10d stubs are untouched by the new kwargs' defaults
    assert StubPolicySession("stub").contract["rpy_convention"] is None
    assert StubPolicySession("stub", action_repr="pose10d").contract["rpy_convention"] is None


def test_identity_normaliser_refusal_applies_to_every_non_pose10d_repr():
    """The training-side trap (upstream get_normalizer's pose10d slice
    leaves the middle columns at scale 1 / offset 0): refused for the
    5-dim AND the 7-dim contract (an identity droll/dpitch column is the
    same defect as an identity dyaw), never for the legacy pose10d whose
    six rot6d columns are identity BY DESIGN."""
    ok5 = (np.full(5, 2.0), np.full(5, -0.5))
    assert identity_normaliser_columns("pos_yaw_width", *ok5) == []
    bad5 = (np.array([2.0, 2.0, 2.0, 1.0, 2.0]), np.array([0.1, 0.1, 0.1, 0.0, 0.1]))
    assert identity_normaliser_columns("pos_yaw_width", *bad5) == [3]
    ok7 = (np.full(7, 2.0), np.full(7, -0.5))
    assert identity_normaliser_columns("pos_rpy_width", *ok7) == []
    bad7 = (np.array([2, 2, 2, 2, 1.0, 1.0, 2]), np.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1]))
    assert identity_normaliser_columns("pos_rpy_width", *bad7) == [4, 5]
    # scale 1 with a non-zero offset is NOT identity (a range that happens to be 2)
    assert identity_normaliser_columns("pos_rpy_width",
                                       np.array([2, 2, 2, 2, 1.0, 2, 2]),
                                       np.array([0, 0, 0, 0, 0.3, 0, 0])) == []
    leg = (np.array([2, 2, 2] + [1.0] * 6 + [2]), np.array([0.1] * 3 + [0.0] * 6 + [0.1]))
    assert identity_normaliser_columns("pose10d", *leg) == []


def test_rpy_convention_of_reads_the_contract_key():
    """``shape_meta.action.rpy_convention`` -> a plain dict of str / list of
    str (OmegaConf containers arrive as dict / list, tuples are lists too),
    None when absent (every 5-dim / pose10d checkpoint), ValueError when
    present but not a mapping. Equality with the station constant is the
    backend's decision (test_policy.py mount check), not this reader's."""
    assert rpy_convention_of({}) is None
    assert rpy_convention_of({"rpy_convention": None}) is None
    got = rpy_convention_of({"rpy_convention": {
        "order": "zyx", "frame": "body_frd_via_R_bt",
        "columns": ("dx", "dy", "dz", "dyaw", "droll", "dpitch", "width")}})
    assert got == dp_policy.RPY_CONVENTION_STATION
    assert isinstance(got["columns"], list)
    other = rpy_convention_of({"rpy_convention": {"order": "xyz", "frame": "world",
                                                  "columns": ["dx"]}})
    assert other != dp_policy.RPY_CONVENTION_STATION and other["order"] == "xyz"
    for bad in ("zyx", 7, ["zyx"]):
        try:
            rpy_convention_of({"rpy_convention": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"rpy_convention_of accepted {bad!r}")
    assert dp_policy.RPY_CONVENTION_STATION["columns"] == [
        "dx", "dy", "dz", "dyaw", "droll", "dpitch", "width"]   # col 3 = dyaw, width = D-1


def test_is_stub_ckpt_and_the_stub_variants():
    assert is_stub_ckpt("stub") and is_stub_ckpt(" STUB ") and is_stub_ckpt("stub7")
    assert is_stub_ckpt("stub_rp") and is_stub_ckpt("Stub_RP")
    assert not is_stub_ckpt("") and not is_stub_ckpt(None)
    assert not is_stub_ckpt("/x/stub.ckpt") and not is_stub_ckpt("stub8")
    assert set(dp_policy.STUB_CKPT_VARIANTS) == {"stub", "stub7", "stub_rp"}
    assert dp_policy.STUB_CKPT_VARIANTS["stub"] == {}
    s = make_policy_session({"ckpt": "stub7"}, None)
    assert isinstance(s, StubPolicySession) and s.contract["action_dim"] == 7
    assert s.ckpt == "stub7" and s.pitch_ramp_deg == 0.0
    s = make_policy_session({"ckpt": "/block.ckpt"}, None, ckpt="STUB_RP")
    assert isinstance(s, StubPolicySession) and s.contract["action_dim"] == 7
    assert s.pitch_ramp_deg == 5.0 and s.roll_ramp_deg == 0.0 and s.ckpt == "stub_rp"
    s.load()
    a, _ = s.predict(_stub_obs(s.contract))
    assert abs(math.degrees(a[-1, 5]) - 5.0) < 1e-5 and np.all(a[:, 4] == 0.0)


def test_backend_mount_check_requires_mount_and_convention_for_pos_rpy_width():
    """backends/policy.py ``_check_mount`` (2026-09-26): a ``pos_rpy_width``
    contract WITHOUT the yaw-axis keys is a mount mismatch (no legacy-
    permissive path -- roll and pitch are labelled through the same R_bt),
    and its ``rpy_convention`` must equal the station constant, columns in
    order. The helper is a staticmethod so it is checked here without a
    worker; the 5-dim / pose10d path is untouched (``_rpy_contract_why`` is
    only consulted for pos_rpy_width)."""
    from rov_gui.backends.policy import PolicyWorker
    from rov_gui.control.policy_frames import R_BT_C3

    why = PolicyWorker._rpy_contract_why
    R = np.asarray(R_BT_C3).tolist()
    good = {"action_repr": "pos_rpy_width", "yaw_axis_cam_tilt_deg": 43.3,
            "yaw_axis_R_frd_cam": R,
            "rpy_convention": dict(dp_policy.RPY_CONVENTION_STATION)}
    assert why(good, R) == ""
    # the 7-dim stub's contract passes as-is
    s7 = StubPolicySession("stub7", action_repr="pos_rpy_width")
    assert why(s7.contract, s7.contract["yaw_axis_R_frd_cam"]) == ""
    no_mount = dict(good, yaw_axis_R_frd_cam=None, yaw_axis_cam_tilt_deg=None)
    assert "yaw_axis" in why(no_mount, None)
    no_tilt = dict(good, yaw_axis_cam_tilt_deg=None)
    assert "yaw_axis" in why(no_tilt, R)
    no_conv = dict(good, rpy_convention=None)
    assert "rpy_convention" in why(no_conv, R)
    wrong_order = dict(good, rpy_convention={
        "order": "zyx", "frame": "body_frd_via_R_bt",
        "columns": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "width"]})
    assert "rpy_convention" in why(wrong_order, R)
    wrong_frame = dict(good, rpy_convention={
        "order": "zyx", "frame": "world",
        "columns": list(dp_policy.RPY_CONVENTION_STATION["columns"])})
    assert "rpy_convention" in why(wrong_frame, R)
    # tuples / OmegaConf-style lists compare equal to the constant
    tup = dict(good, rpy_convention={"order": "zyx", "frame": "body_frd_via_R_bt",
                                     "columns": tuple(dp_policy.RPY_CONVENTION_STATION["columns"])})
    assert why(tup, R) == ""


def test_make_policy_session_picks_the_stub():
    blk = {"ckpt": "stub", "repo": "/nonexistent", "num_inference_steps": 16,
           "eval_transforms": "center", "dataset_fps": 30.0}
    s = make_policy_session(blk, None)
    assert isinstance(s, StubPolicySession)
    assert abs(s.contract["obs_dt_s"] - 2.0 / 30.0) < 1e-12
    # CLI override wins over the block, both ways
    opts = types.SimpleNamespace(policy_ckpt="stub", policy_repo=None)
    s = make_policy_session({"ckpt": "/some/real.ckpt"}, opts)
    assert isinstance(s, StubPolicySession)
    opts = types.SimpleNamespace(policy_ckpt="/other/real.ckpt",
                                 policy_repo="/some/repo")
    s = make_policy_session({"ckpt": "stub", "eval_transforms": "train",
                             "num_inference_steps": 8}, opts)
    assert isinstance(s, DpPolicySession) and not s.ready
    assert s.ckpt == "/other/real.ckpt" and s.repo == "/some/repo"
    assert s.eval_transforms == "train" and s.num_inference_steps == 8
    # a real path is a real session, unloaded, with the block's choices
    s = make_policy_session({"ckpt": "/x.ckpt", "repo": "/r"}, None)
    assert isinstance(s, DpPolicySession) and s.eval_transforms == "center"
    try:
        DpPolicySession("/x.ckpt", "/r", eval_transforms="random")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown eval_transforms accepted")


def test_make_policy_session_explicit_ckpt_wins():
    """2026-09-11 panel picker: ``PolicyWorker.set_ckpt`` passes the pick as
    the explicit ``ckpt=`` kwarg, which beats BOTH ``opts.policy_ckpt`` (the
    removed flag's attribute, still set directly by tests/tools for the stub)
    and the block; ``None`` / ``""`` keep the pre-2026-09-11 precedence
    (test_make_policy_session_picks_the_stub, unchanged)."""
    opts = types.SimpleNamespace(policy_ckpt="/opts/real.ckpt",
                                 policy_repo="/some/repo")
    blk = {"ckpt": "/block/real.ckpt", "repo": "/r"}
    s = make_policy_session(blk, opts, ckpt="stub")
    assert isinstance(s, StubPolicySession) and s.repo == "/some/repo"
    s = make_policy_session({"ckpt": "stub"},
                            types.SimpleNamespace(policy_ckpt="stub",
                                                  policy_repo=None),
                            ckpt="/gui/picked.ckpt")
    assert isinstance(s, DpPolicySession) and not s.ready
    assert s.ckpt == "/gui/picked.ckpt"
    # None / "" fall through: opts, then the block
    assert make_policy_session(blk, opts, ckpt=None).ckpt == "/opts/real.ckpt"
    assert make_policy_session(blk, opts, ckpt="").ckpt == "/opts/real.ckpt"
    assert make_policy_session(blk, None).ckpt == "/block/real.ckpt"
    assert make_policy_session(blk, None, ckpt="/gui/x.ckpt").ckpt == "/gui/x.ckpt"


# ------------------------------------------------------------- upstream import
_UPSTREAM_MODS = ("diffusion_policy",
                  "diffusion_policy.model",
                  "diffusion_policy.model.vision",
                  "diffusion_policy.model.vision.transformer_obs_encoder",
                  "diffusion_policy.model.diffusion",
                  "diffusion_policy.model.diffusion.transformer_for_action_diffusion",
                  "Utils")


def test_the_upstream_import_leaves_utils_alone_and_demotes_the_checkout():
    """A fake UMI checkout whose encoder module does a bare ``import Utils``
    (the FoundationPose/FoundationStereo hazard, see perception/upstream.py):
    after ``_import_upstream`` the bare name must still be the POSE side's,
    the checkout must sit at the END of sys.path, both classes must come
    back, and the forbidden (zarr-touching) modules must not be loaded.
    Then the other order: nobody held the name -> it is not left behind."""
    tmp = Path(tempfile.mkdtemp(prefix="dp_policy_upstream_"))
    repo = tmp / "umi"
    enc_dir = repo / "diffusion_policy" / "model" / "vision"
    den_dir = repo / "diffusion_policy" / "model" / "diffusion"
    enc_dir.mkdir(parents=True)
    den_dir.mkdir(parents=True)
    (repo / "Utils.py").write_text("SIDE = 'umi'\n")
    (enc_dir / "transformer_obs_encoder.py").write_text(
        "import Utils\nclass TransformerObsEncoder:\n    side = 'umi'\n")
    (den_dir / "transformer_for_action_diffusion.py").write_text(
        "class TransformerForActionDiffusion:\n    side = 'umi'\n")
    pose = tmp / "pose"
    pose.mkdir()
    (pose / "Utils.py").write_text("SIDE = 'pose'\n")

    saved = {k: sys.modules.get(k) for k in _UPSTREAM_MODS}
    saved_path = list(sys.path)
    try:
        for k in saved:
            sys.modules.pop(k, None)
        # the REAL checkout may already sit on sys.path (a real-checkpoint
        # test ran first); ``diffusion_policy`` is a namespace package, so
        # both portions would merge and the real one would win — take it
        # off for the duration (restored in finally).
        real = str(Path(dp_policy.DEFAULT_UMI_REPO))
        sys.path[:] = [p for p in sys.path if p != real]
        sys.path.insert(0, str(pose))                      # sam2_live's insert
        pose_utils = importlib.import_module("Utils")
        assert pose_utils.SIDE == "pose"

        Enc, Den = dp_policy._import_upstream(str(repo))
        assert Enc.side == "umi" and Den.side == "umi"
        assert sys.modules["Utils"] is pose_utils           # untouched
        assert sys.path[-1] == str(repo)                    # demoted to the end
        assert sys.path.index(str(repo)) > sys.path.index(str(pose))
        for name in dp_policy._FORBIDDEN_UPSTREAM:
            assert name not in sys.modules
        assert importlib.import_module("Utils").SIDE == "pose"

        # the other order: nobody had the name
        for k in _UPSTREAM_MODS:
            sys.modules.pop(k, None)
        sys.path.remove(str(pose))
        dp_policy._import_upstream(str(repo))
        assert "Utils" not in sys.modules                   # not left behind
        sys.path.insert(0, str(pose))
        assert importlib.import_module("Utils").SIDE == "pose"

        # a checkout that resolves elsewhere is refused
        for k in _UPSTREAM_MODS:
            sys.modules.pop(k, None)
        other = tmp / "other"
        (other / "diffusion_policy" / "model" / "vision").mkdir(parents=True)
        (other / "diffusion_policy" / "model" / "diffusion").mkdir(parents=True)
        (other / "diffusion_policy" / "model" / "vision"
         / "transformer_obs_encoder.py").write_text(
            "class TransformerObsEncoder: side='other'\n")
        (other / "diffusion_policy" / "model" / "diffusion"
         / "transformer_for_action_diffusion.py").write_text(
            "class TransformerForActionDiffusion: side='other'\n")
        sys.path.insert(0, str(other))
        try:
            dp_policy._import_upstream(str(repo))
        except ImportError as e:
            assert "another diffusion_policy" in str(e)
        else:
            raise AssertionError("a shadowing checkout was accepted")
    finally:
        for k in _UPSTREAM_MODS:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
        sys.path[:] = saved_path


# ------------------------------------------------------------- the real thing
_REAL = {"session": None, "reason": None, "utils_before": None}


def _real_session():
    """One loaded session for the GPU tests; the reason it cannot load is
    printed once and every GPU test skips with it."""
    if _REAL["session"] is not None:
        return _REAL["session"]
    if _REAL["reason"] is not None:
        _skip(_REAL["reason"])
    reason = None
    if not Path(dp_policy.DEFAULT_CKPT).is_file():
        reason = f"checkpoint missing: {dp_policy.DEFAULT_CKPT}"
    elif not Path(dp_policy.DEFAULT_UMI_REPO).is_dir():
        reason = f"UMI checkout missing: {dp_policy.DEFAULT_UMI_REPO}"
    else:
        try:
            import torch
            if not torch.cuda.is_available():
                reason = "CUDA not available"
        except ImportError:
            reason = "torch not installed in this interpreter"
    if reason is None:
        _REAL["utils_before"] = sys.modules.get("Utils")
        s = DpPolicySession(eval_transforms="center")
        try:
            s.load()
        except Exception as e:                               # noqa: BLE001
            reason = f"load failed: {type(e).__name__}: {e}"
    if reason is not None:
        _REAL["reason"] = reason
        print(f"\n  [real-checkpoint tests skipped: {reason}]")
        _skip(reason)
    _REAL["session"] = s
    atexit.register(s.close)
    return s


def _real_obs():
    from rov_gui.tools.dp_policy_reference import synthetic_obs
    return synthetic_obs(0)


def test_real_checkpoint_loads_strictly():
    s = _real_session()
    assert s.ready and not s.loading and s.error == ""
    assert s.load_seconds > 0.0 and np.isfinite(s.warmup_ms)
    assert not s._encoder.training and not s._model.training
    assert sys.modules.get("Utils") is _REAL["utils_before"]   # A20
    for name in dp_policy._FORBIDDEN_UPSTREAM:
        assert name not in sys.modules
    assert sys.path[-1] == str(Path(dp_policy.DEFAULT_UMI_REPO))


def test_real_contract_values():
    """The DEPLOYED checkpoint must be the 5-dim retrain (2026-09-07): these
    assertions FAIL, not skip, on the legacy 10-dim one -- a stale
    POLICY_CKPT_DEFAULT is exactly what they exist to catch."""
    from rov_gui.control.policy_frames import R_BT_C3, YAW_AXIS_CAM_TILT_DEG

    c = _real_session().contract
    assert list(c["obs_keys"]) == list(StubPolicySession.OBS_KEYS)
    assert c["obs_horizon"] == 2 and c["action_horizon"] == 16
    assert c["action_dim"] == 5 and c["down_sample_steps"] == 2
    assert c["action_repr"] == "pos_yaw_width" == S.POLICY_ACTION_REPR
    assert c["fps"] == 30.0 and c["fps_source"].startswith("dataset zip")
    assert abs(c["obs_dt_s"] - 1.0 / 15.0) < 1e-12
    assert c["image_shape"] == (3, 224, 224)
    assert c["lowdim_shapes"] == StubPolicySession.LOWDIM_SHAPES
    # the action range the normalizer was fitted on (yaw in rad, width in m)
    rng = c["action_range"]
    assert rng is not None and len(rng["min"]) == 5 == len(rng["max"])
    assert np.all(np.asarray(rng["max"]) > np.asarray(rng["min"]))
    assert abs(rng["max"][3]) < math.pi and abs(rng["min"][3]) < math.pi
    assert 0.0 < rng["min"][4] < rng["max"][4] < 0.2
    # the mount the yaw label was defined on, as the backend compares it
    assert abs(float(c["yaw_axis_cam_tilt_deg"]) - YAW_AXIS_CAM_TILT_DEG) < 1e-9
    R = np.asarray(c["yaw_axis_R_frd_cam"], float)
    assert R.shape == (3, 3) and np.allclose(R, R_BT_C3, atol=1e-6)
    assert isinstance(c["ckpt_epoch"], int) and c["ckpt_epoch"] >= 0
    assert isinstance(c["ckpt_global_step"], int) and c["ckpt_global_step"] >= 0
    assert c["rpy_convention"] is None                   # a 5-dim checkpoint


def test_real_predict_shape_dtype_and_range():
    s = _real_session()
    c = s.contract
    act, ms = s.predict(_real_obs())
    assert act.shape == (16, 5) and act.dtype == np.float32
    assert np.all(np.isfinite(act)) and 0.0 < ms < 5000.0
    # DDIM clip_sample + range normalisation: no column can leave the
    # training [min, max] (input_stats) by more than float rounding
    lo = np.asarray(c["action_range"]["min"], float) - 1e-4
    hi = np.asarray(c["action_range"]["max"], float) + 1e-4
    for j in range(5):
        assert np.all(act[:, j] >= lo[j]) and np.all(act[:, j] <= hi[j]), \
            (j, act[:, j].min(), act[:, j].max(), lo[j], hi[j])
    assert 0.0 < act[:, 4].min() and act[:, 4].max() < 0.2      # width, m


def test_real_action_normalizer_has_no_identity_column():
    """The training-side trap (upstream get_normalizer's pose10d slice
    leaves the middle columns at scale 1 / offset 0): a 5-dim checkpoint
    with one would have a yaw column that was never range-normalised, and
    the loader must have refused it. The deployed one shows none."""
    n = _real_session().normalizer
    scale, offset = (np.asarray(v, float) for v in n.params("action"))
    assert scale.shape == (5,) and offset.shape == (5,)
    ident = (scale == 1.0) & (offset == 0.0)
    assert not ident.any(), (scale, offset)
    assert np.all(scale > 1.0), scale                    # every column is a range


def test_real_two_seeded_predicts_are_identical_in_center_mode():
    s = _real_session()
    assert s.eval_transforms == "center"
    obs = _real_obs()
    a1, _ = s.predict(obs, seed=7)
    a2, _ = s.predict(obs, seed=7)
    assert np.array_equal(a1, a2), float(np.abs(a1 - a2).max())
    a3, _ = s.predict(obs, seed=8)
    assert not np.array_equal(a1, a3)                          # seed matters


def test_real_inference_is_under_500_ms():
    s = _real_session()
    obs = _real_obs()
    ms = [s.predict(obs)[1] for _ in range(3)]
    assert float(np.median(ms)) < 500.0, ms


def test_real_describe_carries_the_run_record_fields():
    s = _real_session()
    s.predict(_real_obs())                    # the warm-up is not counted
    d = s.describe()
    for k in ("ckpt", "ckpt_sha1_head", "cfg_fingerprint", "model_name",
              "num_inference_steps", "timesteps", "eval_transforms",
              "torch", "diffusers", "timm", "normalizer_input_stats",
              "load_s", "n_predict", "infer_ms_p50", "infer_ms_max",
              "contract", "stub", "device", "transforms"):
        assert k in d, k
    assert d["stub"] is False and len(d["ckpt_sha1_head"]) == 40
    assert len(d["cfg_fingerprint"]) == 40
    assert d["timesteps"] == [45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12,
                              9, 6, 3, 0]
    assert d["num_inference_steps"] == 16 and d["model_name"] == "resnet34.a1_in1k"
    st = d["normalizer_input_stats"]
    assert set(st) == {"robot0_eef_pos", "robot0_gripper_width"}
    assert abs(st["robot0_gripper_width"]["min"][0] - 0.0246) < 1e-3
    assert abs(st["robot0_gripper_width"]["max"][0] - 0.088) < 1e-3
    assert len(st["robot0_eef_pos"]["min"]) == 3
    assert d["n_predict"] >= 1 and np.isfinite(d["infer_ms_p50"])
    assert d["transforms"][0].startswith("CenterCrop")
    # the contract rides in describe(), epoch included (the run meta's
    # provenance for WHICH epoch flew)
    assert d["contract"]["action_repr"] == "pos_yaw_width"
    assert isinstance(d["contract"]["ckpt_epoch"], int)
    assert isinstance(d["contract"]["ckpt_global_step"], int)


# ------------------------------------------------------------- runner
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except unittest.SkipTest as e:
            print(f"  skip  {name}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
