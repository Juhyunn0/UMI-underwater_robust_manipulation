#!/usr/bin/env python3
"""test_mpc_tune.py — MPC_Tuned profiles (control/mpc_tune.py, 2026-10-04).

What is pinned, in the order a failure would hurt:
  * a profile is APPLIED only under mpc_tuned (and mpc_tuned_7d, which flies
    that mode): mpc / dobmpc / dobmpc_tuned keep the build weights and the
    config trim, and switching back writes the build baseline into the
    solver — the "rotated W outlives the mode" failure, one level up;
  * a profile is complete or it is not loaded (missing / unknown keys raise);
  * save -> load round-trips to exactly what the station flies without one;
  * the worker refuses a pick while engaged / with a CSV open, and a launch
  * profile that fails to load refuses mpc_tuned engage instead of quietly
    flying hw_mpc.yaml.

    ~/miniforge3/envs/robust/bin/python -m pytest -q rov_gui/tests/test_mpc_tune.py
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import mpc_tune as MT                       # noqa: E402
from rov_gui.control.geometry import MpcConfig                   # noqa: E402

SHIPPED = ROOT / "config" / "hw_mpc.yaml"


def _pc(**over) -> dict:
    d = {"along_scale": 0.25, "cross_scale": 4.0, "q_along": None,
         "q_cross": None, "apply_terminal": True, "split_velocity": False,
         "v_along_scale": 1.0, "v_cross_scale": 1.0}
    d.update(over)
    return d


def _apply(**over) -> dict:
    a = {"Q": [300.0, 300.0, 150.0, 80.0, 80.0, 150.0] + [10.0] * 6,
         "QN": [300.0, 300.0, 150.0, 80.0, 80.0, 150.0] + [10.0] * 6,
         "R": [0.05, 0.05, 0.05, 0.01, 0.01, 0.005],
         "path_cost": _pc(),
         "policy": {"q_scale": 4.0, "along_scale": 3.0},
         "vehicle_net_buoyancy_n": -12.0, "axis_cap": 0.5,
         "axis_slew_per_s": 1.5}
    a.update(over)
    return a


def _rename(p: Path, stem: str) -> Path:
    q = p.with_name(f"{stem}.yaml")
    p.rename(q)
    return q


def _write(tmp, apply=None, **top) -> Path:
    d = {"schema": MT.SCHEMA, "name": "t", "apply": apply or _apply()}
    d.update(top)
    p = Path(tmp) / f"{d['name']}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


# ------------------------------------------------------------------ the file
def test_the_shipped_default_profile_loads_and_is_named_by_the_config():
    cfg = MpcConfig.load(SHIPPED)
    assert cfg.mpc_tuned_profile, "hw_mpc.yaml names no default profile"
    prof = MT.load_profile(cfg.mpc_tuned_profile)
    assert Path(prof.path).parent == MT.PROFILE_DIR
    assert len(prof.sha1) == 40 and prof.description


def test_a_profile_is_complete_or_it_is_not_loaded(tmp_path):
    for k in MT.APPLY_KEYS:
        a = _apply()
        del a[k]
        with pytest.raises(ValueError, match="missing"):
            MT.load_profile(_write(tmp_path, a))
    with pytest.raises(ValueError, match="unknown"):
        MT.load_profile(_write(tmp_path, _apply(q_scale=4.0)))       # wrong level
    with pytest.raises(ValueError, match="unknown"):
        MT.load_profile(_write(tmp_path, _apply(path_cost=_pc(cross_sale=4.0))))
    with pytest.raises(ValueError, match="path_cost: missing"):
        # a partial split would be filled from DEFAULT_TUNE — never written
        MT.load_profile(_write(tmp_path, _apply(path_cost={"along_scale": 1.0})))
    with pytest.raises(ValueError, match="file name"):
        # a copied file whose name: still says the original
        MT.load_profile(_rename(_write(tmp_path), "copied"))
    with pytest.raises(ValueError, match="policy"):
        MT.load_profile(_write(tmp_path, _apply(policy={"q_scale": 4.0})))
    with pytest.raises(ValueError, match="top-level"):
        MT.load_profile(_write(tmp_path, applied={}))
    with pytest.raises(ValueError, match="schema"):
        MT.load_profile(_write(tmp_path, schema=99))
    with pytest.raises(FileNotFoundError):
        MT.load_profile(tmp_path / "nope.yaml")


def test_bad_values_raise_and_null_along_is_the_controllers_own(tmp_path):
    zero_xy = [0.0, 0.0, 150.0, 80.0, 80.0, 150.0] + [10.0] * 6
    for over in ({"Q": [1.0] * 11}, {"R": [0.05] * 5 + [0.0]},
                 # a zero pose weight: no hold, and q_along 0 -> 0/0 in meta
                 {"Q": zero_xy}, {"QN": zero_xy},
                 {"vehicle_net_buoyancy_n": 16.0}, {"axis_cap": 0.0},
                 {"axis_cap": 1.5}, {"axis_slew_per_s": -1.0},
                 # 0 = NO slew limit: a profile may not switch it off
                 {"axis_slew_per_s": 0.0},
                 {"policy": {"q_scale": 0.0, "along_scale": None}},
                 {"policy": {"q_scale": 1.0, "along_scale": -1.0}}):
        with pytest.raises(ValueError):
            MT.load_profile(_write(tmp_path, _apply(**over)))
    p = MT.load_profile(_write(tmp_path, _apply(
        policy={"q_scale": 2.0, "along_scale": None})))
    assert p.apply["policy"] == {"q_scale": 2.0, "along_scale": None}
    # the split is resolved to the full key set (defaults filled in)
    assert set(p.apply["path_cost"]) == set(
        __import__("rov_gui.control.path_cost", fromlist=["TUNE_KEYS"]).TUNE_KEYS)


def test_no_profile_names():
    for n in (None, "", "none", "None", "null", "hw_mpc.yaml"):
        assert MT.is_no_profile(n)
    assert not MT.is_no_profile("20261004_default")


def _dobmpc_P():
    try:
        from rov_gui.control.mpc_bridge import apply_plant_overrides, import_dobmpc
        P = import_dobmpc("heavy_gripper")["P"]
    except Exception as e:                                       # noqa: BLE001
        pytest.skip(f"no dobmpc here ({type(e).__name__}: {e})")
    cfg = MpcConfig.load(SHIPPED)
    apply_plant_overrides(P, cfg.plant)
    return cfg, P


def test_save_round_trips_to_what_the_station_flies_without_one(tmp_path):
    cfg, P = _dobmpc_P()
    d = MT.snapshot(cfg, P, name="rt", description="x", config_path=SHIPPED)
    path = MT.write_profile(d, tmp_path / "rt.yaml")
    prof = MT.load_profile(path)
    want = MT.current_apply(cfg, P)
    assert prof.apply == want
    assert MT.profile_warnings(prof, P=P, cfg=cfg) == []
    with pytest.raises(FileExistsError):
        MT.write_profile(d, path)
    # the record carries the hw_mpc.yaml blocks it was taken from
    assert prof.record["hw_mpc_yaml"]["vehicle_net_buoyancy_n"] == \
        cfg.raw["vehicle_net_buoyancy_n"]


def test_a_different_solver_or_vehicle_is_a_warning_not_a_silent_pass(tmp_path):
    cfg, P = _dobmpc_P()
    d = MT.snapshot(cfg, P, name="w", config_path=SHIPPED)
    d["compare"]["linear_damping"] = [4.03, 6.22, 5.18, 0.07, 0.07, 0.07]
    d["compare"]["axis_gain"]["surge_n"] = 30.0
    d["apply"]["axis_cap"] = 0.9
    d["apply"]["axis_slew_per_s"] = float(cfg.axis_slew_per_s) * 4.0
    d["apply"]["vehicle_net_buoyancy_n"] = float(cfg.vehicle_net_buoyancy_n) + 5.0
    del d["compare"]["dt_s"]
    prof = MT.load_profile(MT.write_profile(d, tmp_path / "w.yaml"))
    w = " | ".join(MT.profile_warnings(prof, P=P, cfg=cfg))
    assert "linear_damping" in w and "surge_n" in w and "axis_cap 0.9" in w
    assert "axis_slew_per_s" in w                    # looser than the station
    assert "vehicle_net_buoyancy_n" in w             # the peg trap
    assert "compare.dt_s missing" in w               # never a quiet OK


def test_save_from_a_held_profile_saves_what_mpc_tuned_flies(tmp_path):
    """The config names a profile -> mpc_tuned flies ITS apply, so that is
    what a snapshot must hold, not hw_mpc.yaml's ignored blocks."""
    cfg, P = _dobmpc_P()
    base = MT.load_profile(_write(tmp_path, _apply(
        path_cost=_pc(along_scale=0.7), vehicle_net_buoyancy_n=-3.0)))
    d = MT.snapshot(cfg, P, name="copy", config_path=SHIPPED, base=base)
    prof = MT.load_profile(MT.write_profile(d, tmp_path / "copy.yaml"))
    assert prof.apply == base.apply
    assert prof.source["apply_from"] == {"profile": "t", "sha1": base.sha1}
    assert prof.apply != MT.current_apply(cfg, P)


def test_the_profile_slew_can_only_tighten():
    assert MT.effective_slew(0.7, 1.5) == 0.7
    assert MT.effective_slew(4.0, 1.5) == 1.5
    assert MT.effective_slew(4.0, 0.0) == 4.0        # station has none


# ------------------------------------------------------- the bridge (no solver)
class _Solver:
    def __init__(self):
        self.W = {}

    def cost_set(self, k, field, W, api="new"):
        assert field == "W"
        self.W[k] = np.array(W, float)


class _Nmpc:
    N = 3

    def __init__(self):
        self.solver = _Solver()

    def reset(self):
        pass


def _bare(mode="mpc", trim=0.1):
    """A HwDobMpc with no acados build: the fields the profile path touches."""
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.path_cost import PathFrameWeights

    cfg, P = _dobmpc_P()
    cfg = copy.deepcopy(cfg)
    cfg.vehicle_net_buoyancy_n = trim
    obj = HwDobMpc.__new__(HwDobMpc)
    obj.P, obj.cfg, obj.nmpc = P, cfg, _Nmpc()
    obj.solver_kind = "acados"
    obj._mode, obj.dob, obj.tuned = mode, mode.startswith("dob"), mode.endswith("_tuned")
    obj._w_tuned = False
    obj._plan_q_scale = obj._plan_q_scale_flown = 1.0
    obj._plan_along_scale = obj._plan_along_scale_flown = None
    obj._tune_profile, obj._tune_warnings = None, []
    obj._Q_build = np.asarray(P.MPC_Q, float).copy()
    obj._R_build = np.asarray(P.MPC_R, float).copy()
    obj._QN_build = np.asarray(P.MPC_QN, float).copy()
    obj._wt_cfg_tune = cfg.mpc_tuned
    obj._wt = PathFrameWeights(obj._Q_build, obj._R_build, obj._QN_build,
                               obj._wt_cfg_tune)
    obj.vehicle_net_buoyancy_n = trim
    obj.w_trim = HwDobMpc.heave_trim_wrench(float(P.NET_BUOYANCY), trim)
    return obj


def test_a_profile_applies_under_mpc_tuned_only_and_leaves_no_trace(tmp_path):
    a = _apply(Q=[600.0, 600.0, 150.0, 80.0, 80.0, 150.0] + [20.0] * 6,
               path_cost=_pc(along_scale=1.0, cross_scale=2.0),
               policy={"q_scale": 2.5, "along_scale": None},
               vehicle_net_buoyancy_n=-8.0, axis_cap=0.4,
               axis_slew_per_s=0.7)
    prof = MT.load_profile(_write(tmp_path, a))
    obj = _bare("mpc", trim=0.1)
    W_build = np.array(obj._wt.W_base)
    trim_cfg = obj.w_trim.copy()
    obj.set_tune_profile(prof)
    # HELD under mpc, not applied: build weights, config trim, no caps
    assert obj.tune_profile is prof and not obj.tune_profile_active
    assert np.array_equal(obj._wt.W_base, W_build)
    assert np.array_equal(obj.w_trim, trim_cfg)
    assert obj.axis_cap is None and obj.axis_slew_per_s is None
    assert obj.policy_follow(4.0, 3.0) == (4.0, 3.0)
    # mpc_tuned: everything in apply
    obj.mode = "mpc_tuned"
    assert obj.tune_profile_active
    assert np.allclose(obj._wt.Q, a["Q"]) and obj._wt.q_along == 1.0 * 600.0
    assert obj._wt.q_cross == 2.0 * 600.0
    assert obj.vehicle_net_buoyancy_n == -8.0
    assert obj.w_trim[2] == pytest.approx(float(obj.P.NET_BUOYANCY) + 8.0)
    assert obj.axis_cap == 0.4 and obj.axis_slew_per_s == 0.7
    assert obj.policy_follow(4.0, 3.0) == (2.5, None)
    # ...and the profile's BASELINE is in the solver even with no path plan
    assert np.array_equal(obj.nmpc.solver.W[0], obj._wt.W_base)
    assert obj.nmpc.solver.W[0][0, 0] == 600.0
    m = obj.tune_meta()
    assert m["active"] and m["profile"] == "t" and m["sha1"] == prof.sha1
    # dobmpc_tuned is NOT mpc_tuned: the config's split and build weights
    obj.mode = "dobmpc_tuned"
    assert not obj.tune_profile_active
    assert np.array_equal(obj._wt.W_base, W_build)
    assert obj._wt.tune["along_scale"] == MpcConfig.load(SHIPPED).mpc_tuned["along_scale"]
    # back to mpc: the BUILD baseline is what the solver holds again
    obj.mode = "mpc_tuned"
    obj.mode = "mpc"
    assert np.array_equal(obj.nmpc.solver.W[0], W_build)
    assert np.array_equal(obj.w_trim, trim_cfg)
    assert obj.tune_meta()["active"] is False
    # dropping the profile under mpc_tuned restores hw_mpc.yaml's values
    obj.mode = "mpc_tuned"
    obj.set_tune_profile(None)
    assert obj.tune_profile is None and not obj.tune_profile_active
    assert np.array_equal(obj._wt.W_base, W_build)
    assert obj.vehicle_net_buoyancy_n == 0.1


def test_the_policy_along_scale_restores_to_the_profiles_split(tmp_path):
    prof = MT.load_profile(_write(tmp_path, _apply(
        path_cost=_pc(along_scale=0.5, cross_scale=3.0))))
    obj = _bare("mpc_tuned")
    obj.set_tune_profile(prof)
    q_xy = obj._wt.q_xy
    obj.set_plan_along_scale(3.0)
    assert obj._wt.q_along == pytest.approx(3.0 * q_xy)
    assert obj._wt.q_cross == pytest.approx(3.0 * q_xy)       # cross untouched
    obj.set_plan_along_scale(None)                             # STOP
    assert obj._wt.q_along == pytest.approx(0.5 * q_xy)        # the PROFILE's own


def test_a_profile_that_does_not_fit_the_solver_is_refused_and_nothing_changes(tmp_path):
    prof = MT.load_profile(_write(tmp_path, _apply()))
    obj = _bare("mpc_tuned")
    obj.set_tune_profile(prof)
    bad = copy.deepcopy(prof)
    bad.apply["R"] = [0.05] * 4
    with pytest.raises(ValueError, match="sizes"):
        obj.set_tune_profile(bad)
    assert obj.tune_profile is prof                           # still the old one
    obj.solver_kind = "ipopt"
    with pytest.raises(ValueError, match="acados"):
        obj.set_tune_profile(prof)


def test_a_pick_that_fails_mid_apply_rolls_back_to_the_previous_profile(tmp_path):
    """Review 2026-10-04: assign-then-refresh left a new profile HELD with a
    half-built W when the solver write raised. Now it is all or nothing."""
    good = MT.load_profile(_write(tmp_path, _apply()))
    obj = _bare("mpc_tuned")
    obj.set_tune_profile(good)
    W0, trim0 = np.array(obj._wt.W_base), obj.w_trim.copy()
    other = copy.deepcopy(good)
    other.name = "other"
    other.apply["Q"] = [900.0] * 6 + [10.0] * 6
    other.apply["vehicle_net_buoyancy_n"] = 3.0

    class Boom(_Solver):
        def cost_set(self, k, field, W, api="new"):
            raise RuntimeError("acados said no")

    obj.nmpc.solver = Boom()
    with pytest.raises(RuntimeError):
        obj.set_tune_profile(other)
    assert obj.tune_profile is good
    assert np.array_equal(obj._wt.W_base, W0) and np.array_equal(obj.w_trim, trim0)
    with pytest.raises(RuntimeError):
        obj.mode = "mpc"                              # the setter rolls back too
    assert obj.mode == "mpc_tuned" and obj.tune_profile_active
    assert np.array_equal(obj._wt.W_base, W0)


# --------------------------------------------------- closed loop (real solver)
def test_the_real_solver_flies_the_profile_and_mpc_is_not_contaminated(tmp_path):
    """One real acados bridge: the profile's Q changes the mpc_tuned wrench,
    and plain mpc afterwards is bit-for-bit what it was before."""
    try:
        from rov_gui.control.mpc_bridge import HwDobMpc
        c = HwDobMpc("mpc", MpcConfig(), log=lambda m: None)
    except Exception as e:                                       # noqa: BLE001
        pytest.skip(f"no acados build here ({type(e).__name__}: {e})")
    eta = np.array([0.05, -0.03, 0.82, 0.0, 0.0, 0.1])
    nu = np.zeros(6)

    def solve(mode):
        c.mode = mode
        c.reset()
        c.set_target_ned([0.0, 0.0, 0.8], 0.0)
        u, _ = c.step(eta, nu, np.zeros(6), 0.0)
        return np.asarray(u, float)

    u_mpc0 = solve("mpc")
    u_tuned0 = solve("mpc_tuned")
    prof = MT.load_profile(_write(tmp_path, _apply(
        Q=[1200.0, 1200.0, 600.0, 80.0, 80.0, 150.0] + [10.0] * 6,
        QN=[1200.0, 1200.0, 600.0, 80.0, 80.0, 150.0] + [10.0] * 6,
        vehicle_net_buoyancy_n=0.0)))
    c.set_tune_profile(prof)
    u_tuned1 = solve("mpc_tuned")
    assert np.abs(u_tuned1[:3]).max() > np.abs(u_tuned0[:3]).max() + 0.5, (u_tuned0, u_tuned1)
    assert c.meta()["weights_source"].startswith("mpc_tune profile")
    assert c.meta()["Q"][0] == 1200.0
    u_mpc1 = solve("mpc")
    assert np.abs(u_mpc1 - u_mpc0).max() < 1e-6, (u_mpc0, u_mpc1)
    assert c.meta()["Q"][0] == 300.0 and c.meta()["mpc_tune"]["active"] is False


# ----------------------------------------------------------------- the worker
def _worker_with_real_bridge(tmp, launch=None):
    from rov_gui.bus import DataBus
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.workers import MpcWorker
    from rov_gui.tests.test_control import _app, _test_opts

    _app()
    try:
        HwDobMpc("mpc", MpcConfig(), log=lambda m: None)
    except Exception as e:                                       # noqa: BLE001
        pytest.skip(f"no acados build here ({type(e).__name__}: {e})")
    Opts = _test_opts(tmp)
    Opts.mpc_tune = launch
    w = MpcWorker(DataBus(), Opts(),
                  controller_factory=lambda cfg: HwDobMpc("mpc_tuned", cfg,
                                                          log=lambda m: None))
    w.setup()
    return w


def test_the_worker_holds_refuses_and_reports_a_pick(tmp_path):
    w = _worker_with_real_bridge(tmp_path)
    try:
        assert w._tune_status() == ("", "", "", "")
        p = _write(tmp_path, _apply(vehicle_net_buoyancy_n=-3.0))
        w.set_mpc_tune(str(p))
        name, path, st, _note = w._tune_status()
        assert name == "t" and path == str(p) and st in ("ok", "warn")
        assert w._mpc_ctrl.tune_profile.name == "t"
        # refused while engaged: the held profile is unchanged
        w.engaged = True
        w.set_mpc_tune("none")
        assert w._mpc_ctrl.tune_profile.name == "t"
        assert w._tune_status()[2] == "refused" and "engaged" in w.reason
        w.engaged = False
        # a bad file: refused, still held
        w.set_mpc_tune(str(tmp_path / "missing.yaml"))
        assert w._mpc_ctrl.tune_profile.name == "t"
        w.set_mpc_tune("none")
        assert w._mpc_ctrl.tune_profile is None
    finally:
        w.teardown()


def test_a_launch_profile_that_fails_refuses_mpc_tuned_instead_of_flying_the_yaml(tmp_path):
    w = _worker_with_real_bridge(tmp_path, launch="no_such_profile_xyz")
    try:
        assert w._tune_error and w._mpc_ctrl.tune_profile is None
        assert w._tune_status()[2] == "error"
        w.cfg.mode = "mpc_tuned"
        assert "MPC_Tuned profile failed" in w._engage_refusal()
        w.cfg.mode = "mpc"
        assert "MPC_Tuned profile" not in w._engage_refusal()
        # a good pick clears it
        w.set_mpc_tune(str(_write(tmp_path, _apply())))
        assert not w._tune_error
    finally:
        w.teardown()


# ------------------------------------------------------------------ the panel
def test_the_tune_row_shows_for_mpc_tuned_only_and_reflects_the_worker():
    from rov_gui.state import MpcStatus
    from rov_gui.tests.test_control import _app
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    w.show()
    try:
        for low, shown in (("pid", False), ("mpc", False), ("mpc_tuned", True),
                           ("mpc_tuned_7d", True), ("dobmpc_tuned", False)):
            w.set_mode_default(low)
            assert all(x.isVisible() == shown for x in w.tune_widgets), low
        w.set_mode_default("mpc_tuned")
        w.add_status(MpcStatus(mode="mpc_tuned"))
        assert w.tune_name.text() == "hw_mpc.yaml" and w.tune_state.text() == ""
        w.add_status(MpcStatus(mode="mpc_tuned", mpc_tune="20261004_default",
                               mpc_tune_path="/x/20261004_default.yaml",
                               mpc_tune_state="warn",
                               mpc_tune_note="linear_damping [1]→[2]"))
        assert w.tune_name.text() == "20261004_default"
        assert w.tune_state.text() == "WARN"
        assert "linear_damping" in w.tune_state.toolTip()
        assert w.current_tune() == "/x/20261004_default.yaml"
        # a pick shows as a REQUEST until the worker answers
        sent = []
        w.mpc_tune_requested.connect(sent.append)
        w._apply_tune_choice("/x/other.yaml")
        assert sent == ["/x/other.yaml"] and w.tune_name.text() == "→ other"
        w.add_status(MpcStatus(mode="mpc_tuned", mpc_tune="20261004_default",
                               mpc_tune_path="/x/20261004_default.yaml",
                               mpc_tune_state="refused",
                               mpc_tune_note="not while engaged"))
        assert w.tune_name.text() == "20261004_default"
        assert w.tune_state.text() == "not while engaged"
        # frozen while engaged / CSV open
        w.add_status(MpcStatus(mode="mpc_tuned", engaged=True))
        assert not w.btn_tune.isEnabled()
        w.add_status(MpcStatus(mode="mpc_tuned", csv_open=True))
        assert not w.btn_tune.isEnabled()
        w.add_status(MpcStatus(mode="mpc_tuned"))
        assert w.btn_tune.isEnabled()
    finally:
        w.close()
        w.deleteLater()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
