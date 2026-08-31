#!/usr/bin/env python3
"""
test_plot_runs.py — the offline replotter (rov_gui/tools/plot_runs.py), on
synthetic run folders. No Qt, no hardware, no recorded session needed.

    ~/miniforge3/envs/robust/bin/python -m pytest rov_gui/tests/test_plot_runs.py -v
    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_plot_runs.py

What is worth testing here is not "does matplotlib draw", it is the three
places where a figure can be quietly WRONG:

* the datum lift — a run's CSV is in its own ENGAGE frame, and if the rigid
  transform back into the tag map is off, every figure still looks like a
  plausible square, just in the wrong part of the pool. Checked against the
  same arithmetic MpcWorker._to_map_xy does, with a fixture whose answer is
  known in closed form.
* the axis box — the whole point of the tool is that the limits never move.
  Checked by deriving the box the way geometry.pool_from_tags does and by
  asserting two different runs get the same one.
* the reporting — a date with no data, a CSV with no rows and a trajectory
  that never started must each come back as a NAMED reason, not as a missing
  file the operator discovers later.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rov_gui.tools import plot_runs as PR

HEADER = ("t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap,rz,ryaw_deg,t_traj,mode,"
          "engaged,traj_on,solver,solver_status,solve_ms,n_tags,pnp_rms_px,"
          "tag_age_s,imu_age_s,z_src,ambig,w0,w1,w2,w3,w4,w5,uX,uY,uZ,uK,uM,"
          "uN,ax_surge,ax_sway,ax_heave,ax_yaw,nis,pwm_dev_us,e_along,e_cross\n")

P0 = (-0.5, 1.25, -1.0)          # the engage pose in TAG coordinates
YAW0_DEG = 30.0


def _meta(ctrl="mpc", datum=True, traj=True, lead=0.1) -> dict:
    m = {
        "schema_version": 1,
        "controller": {"type": ctrl, "solver": "acados", "ctrl_hz": 20.0},
        "trajectory": ({"kind": "square", "size": 1.0, "size_y": 1.0,
                        "speed": 0.05, "laps": 2, "origin_tag": 37}
                       if traj else None),
        "reference_clock": {"path_following": True, "path_lead_m": lead},
        "hardware": {"tag_map_sha1": "deadbeef", "geometry": "floor"},
    }
    if datum:
        m["hardware"]["datum_tag_frame"] = {"p0": list(P0),
                                            "yaw0_deg": YAW0_DEG}
    return m


def _write_run(day: Path, run: str, stem: str, rows: list[str], meta: dict):
    d = day / run
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.csv").write_text(HEADER + "".join(rows), encoding="utf-8")
    (d / f"{stem}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def _square_rows(n_per_side=25, laps=2, offset=0.03, traj_on=True):
    """A square flown `offset` metres OUTSIDE its reference, in datum FLU.

    A constant outward offset makes the expected cross-track RMS exactly
    `offset`, which is what the statistics assertion leans on.
    """
    rows, t = [], 0.0
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for lap in range(laps):
        for i in range(4):
            ax_, ay = corners[i]
            bx, by = corners[(i + 1) % 4]
            for k in range(n_per_side):
                f = k / n_per_side
                rx, ry = ax_ + (bx - ax_) * f, ay + (by - ay) * f
                # outward normal of a CCW square traversal
                nx, ny = (by - ay), -(bx - ax_)
                nn = math.hypot(nx, ny)
                px, py = rx + offset * nx / nn, ry + offset * ny / nn
                rows.append(
                    f"{t:.4f},{px:.5f},{py:.5f},0.00000,{rx:.5f},{ry:.5f},"
                    f"0.000,0.000,{lap},0.00000,0.000,{t:.4f},mpc,1,"
                    f"{1 if traj_on else 0},acados,0,1.00,8,1.0,0.05,0.05,"
                    f"pressure,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0,0,nan,nan\n")
                t += 0.05
    return rows


def _cfg(base: Path, **over) -> dict:
    cfg = PR.deep_merge(PR.DEFAULTS, {"base": str(base), "dates": ["20260814"]})
    return PR.deep_merge(cfg, over)


def _specs(cfg: dict, base: Path):
    return PR.resolve_dates(cfg, base)[0]


# ------------------------------------------------------------------ the lift
def test_a_circle_run_captions_its_radius_and_still_draws_one_lap():
    """The offline figure reads the mission out of meta.json, and a circle's
    `trajectory` block carries `radius` where a rectangle carries
    `size`/`size_y` (meta schema 4). Two things have to survive that: the
    caption must name the shape it actually was, and `ref_xy`'s single-lap
    slicing — which exists so five dashed copies of the reference do not fill
    each other in and render solid — must still find one lap on a CLOSED
    constant-curvature curve.
    """
    from rov_gui.tools import plot_runs as PR

    meta = _meta()
    meta["trajectory"] = {"kind": "circle", "radius": 0.5, "speed": 0.05,
                          "laps": 2, "origin_tag": 37}
    cap = PR._traj_line(meta)
    assert "CIRCLE" in cap and "r 0.50 m" in cap, cap
    assert "×" not in cap, f"a circle captioned with rectangle sides: {cap}"

    # two laps of a circle whose tag is its min-x point, flown exactly
    rows, t = [], 0.0
    n = 60
    for lap in range(2):
        for k in range(n):
            th = 2 * math.pi * k / n
            rx, ry = 0.5 * (1 - math.cos(th)), 0.5 * math.sin(th)
            rows.append(
                f"{t:.4f},{rx:.5f},{ry:.5f},0.00000,{rx:.5f},{ry:.5f},"
                f"0.000,0.000,{lap},0.00000,0.000,{t:.4f},mpc,1,1,acados,0,"
                f"1.00,8,1.0,0.05,0.05,pressure,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                f"0,0,0,0.0,0,nan,nan\n")
            t += 0.05
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _write_run(base / "20260814", "2055", "mpc_205520", rows, meta)
        runs, skips = PR.discover(base, [("20260814", "")], _cfg(base))
        assert len(runs) == 1 and not skips, skips
        rx, ry = runs[0].ref_xy(True)
        assert rx.size == n, f"{rx.size} points drawn, expected one lap of {n}"
        assert runs[0].ref_xy(False)[0].size == 2 * n
        # ...and what is drawn is a circle, not a chord across one. The
        # tolerance is the CSV's own 5-decimal rounding, not a fudge.
        cx, cy = rx.mean(), ry.mean()
        rad = np.hypot(rx - cx, ry - cy)
        assert abs(rad.mean() - 0.5) < 1e-4 and rad.std() < 1e-5, rad.std()
        # THE requirement, stated in a frame-independent way so the map lift
        # cannot hide it: the run STARTS on the rim (one radius from the
        # centre), which is what "the tag is the bottom of the circle, not its
        # centre" means once the figure is in tag-map coordinates.
        assert abs(math.hypot(rx[0] - cx, ry[0] - cy) - 0.5) < 1e-4


def test_datum_lift_matches_the_controller_arithmetic():
    """_to_map must be MpcWorker._to_map_xy composed with the FLU mirror."""
    px, py = np.array([0.4, -0.9]), np.array([0.7, 0.2])
    x, y = PR._to_map(px, py, (P0[0], P0[1], math.radians(YAW0_DEG)))
    c, s = math.cos(math.radians(YAW0_DEG)), math.sin(math.radians(YAW0_DEG))
    xn, yn = px, -py                                   # workers._flu_of_ned
    want_x = P0[0] + c * xn - s * yn                   # workers._to_map_xy
    want_y = P0[1] + s * xn + c * yn
    assert np.allclose(x, want_x) and np.allclose(y, want_y)


def test_datum_lift_puts_the_engage_pose_at_the_datum():
    """(0,0) in the CSV is, by definition, the engage pose in the tag map."""
    x, y = PR._to_map(np.array([0.0]), np.array([0.0]),
                      (P0[0], P0[1], math.radians(YAW0_DEG)))
    assert abs(x[0] - P0[0]) < 1e-12 and abs(y[0] - P0[1]) < 1e-12


def test_lift_is_rigid():
    """Distances survive it — a 1 m square stays a 1 m square."""
    a = PR._to_map(np.array([0.0, 1.0]), np.array([0.0, 0.0]),
                   (P0[0], P0[1], math.radians(YAW0_DEG)))
    d = math.hypot(a[0][1] - a[0][0], a[1][1] - a[1][0])
    assert abs(d - 1.0) < 1e-12


# ------------------------------------------------------------- the error split
def test_along_cross_signs():
    """along + = behind the target, cross + = left of travel (workers.py)."""
    R = np.column_stack([np.linspace(0, 1, 50), np.zeros(50)])   # due +x
    P = R + np.array([-0.1, 0.0])                                # 10 cm behind
    along, cross = PR._along_cross(P, R)
    assert along[10] > 0.09 and abs(cross[10]) < 1e-9
    P = R + np.array([0.0, -0.05])          # in NED, -y of a +x heading = LEFT
    along, cross = PR._along_cross(P, R)
    assert cross[10] > 0.049 and abs(along[10]) < 1e-9


def test_statistics_from_a_known_offset():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _write_run(base / "20260814", "2055", "mpc_205520",
                   _square_rows(offset=0.03), _meta())
        runs, skips = PR.discover(base, [("20260814", "")], _cfg(base))
        assert len(runs) == 1 and not skips
        st = runs[0].stats
        # e_cross is absent from this fixture, so the tangent reconstruction
        # is what runs; the corners are the only places it cannot be exact.
        assert 0.028 < st["cross_rms_cm"] / 100.0 < 0.032
        assert st["path_following"] and abs(st["lead_cm"] - 10.0) < 1e-6
        assert runs[0].ctrl == "mpc" and runs[0].ctrl_label == "MPC · acados"


# ------------------------------------------------------------------- the box
def test_pool_box_reproduces_the_stations_derivation():
    """The drawn box must BE the box the station draws.

    Checked against the real ``NavConfig.pool_from_tags`` rather than against
    a copy of its arithmetic, so that changing the derivation in geometry.py
    and forgetting this tool shows up here. That import needs the control
    package's dependencies; where they are missing the check falls back to
    re-deriving from the YAML, which is weaker but still catches a wrong
    margin or a wrong map.
    """
    box, prov = PR.resolve_pool(PR.DEFAULTS, [])
    assert "tag_map_full" in prov and "pool_margin_m" in prov
    try:
        from rov_gui.control.geometry import NavConfig
        cfg = NavConfig.load(PR.REPO / "config" / "hw_nav.yaml")
        want = cfg.pool_from_tags(cfg.make_tag_map())
    except Exception as e:                                       # noqa: BLE001
        print(f"        (geometry.py unavailable: {type(e).__name__}) ", end="")
        import yaml
        nav = yaml.safe_load((PR.REPO / "config" / "hw_nav.yaml").read_text())
        P, n = PR._tag_map_extent(PR.REPO / nav["tag_map"])
        pad = float(nav["tag_size_m"]) / 2.0 + float(nav["pool_margin_m"])
        assert n > 0
        want = {"x": [P[:, 0].min() - pad, P[:, 0].max() + pad],
                "y": [P[:, 1].min() - pad, P[:, 1].max() + pad]}
    for k in ("x", "y"):
        assert np.allclose(box[k], want[k], atol=1e-12), (k, box[k], want[k])


def test_pool_box_is_the_same_for_every_run():
    """The one property the whole tool exists for: the axes never move."""
    box, _ = PR.resolve_pool(PR.DEFAULTS, [])
    again, _ = PR.resolve_pool(PR.DEFAULTS, [Path("/nonexistent")])
    assert box == again
    explicit = PR.deep_merge(PR.DEFAULTS,
                             {"pool": {"source": "config", "x": [-1.0, 1.0],
                                       "y": [-2.0, 2.0]}})
    box3, prov3 = PR.resolve_pool(explicit, [])
    assert box3 == {"x": [-1.0, 1.0], "y": [-2.0, 2.0]} and "traj_plots" in prov3


# -------------------------------------------------------------- the reporting
def test_a_date_with_no_data_is_named_not_silently_dropped():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        (base / "20260814" / "2004").mkdir(parents=True)
        (base / "20260814" / "2004" / "events.log").write_text("nothing\n")
        runs, skips = PR.discover(base, [("20260814", "")], _cfg(base))
        assert not runs and len(skips) == 1
        assert "no run folder holds a controller CSV" in skips[0]

        specs, missing = PR.resolve_dates(_cfg(base, dates=["20260101"]), base)
        assert specs == [] and missing == ["20260101"]
        specs, _ = PR.resolve_dates(_cfg(base, dates="all"), base)
        assert specs == [("20260814", "")]
        specs, _ = PR.resolve_dates(_cfg(base, dates=["20260814_2120"]), base)
        assert specs == [("20260814", "2120")]
        _, missing = PR.resolve_dates(_cfg(base, dates=["not-a-date"]), base)
        assert missing and "not a YYYYMMDD" in missing[0]


def test_header_only_csv_and_never_started_trajectory_report_why():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        day = base / "20260814"
        _write_run(day, "2004", "mpc_200416", [], _meta())
        _write_run(day, "2014", "mpc_201440",
                   _square_rows(laps=1, traj_on=False), _meta(traj=False))
        # `segment: traj` is the strict reading — the trajectory or nothing.
        runs, skips = PR.discover(base, [("20260814", "")],
                                  _cfg(base, segment="traj"))
        assert not runs and len(skips) == 2
        assert any("no rows" in s for s in skips)
        assert any("min_points" in s and "trajectory running" in s
                   for s in skips)
        # ...and the header-only CSV is unplottable under ANY segment rule.
        runs, skips = PR.discover(base, [("20260814", "")], _cfg(base))
        assert [r.run_dir.name for r in runs] == ["2014"]
        assert len(skips) == 1 and "no rows" in skips[0]


def test_a_run_with_no_datum_is_refused_in_map_frame_and_kept_in_datum_frame():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _write_run(base / "20260814", "2055", "mpc_205520",
                   _square_rows(), _meta(datum=False))
        runs, skips = PR.discover(base, [("20260814", "")], _cfg(base))
        assert not runs and "datum_tag_frame" in skips[0]
        runs, skips = PR.discover(base, [("20260814", "")], _cfg(base, frame="datum"))
        assert len(runs) == 1 and not skips


def test_controller_filter():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        day = base / "20260814"
        _write_run(day, "2055", "mpc_205520", _square_rows(), _meta("mpc"))
        _write_run(day, "2120", "mpc_212021", _square_rows(), _meta("pid"))
        runs, _ = PR.discover(base, [("20260814", "")],
                              _cfg(base, controllers=["pid"]))
        assert [r.ctrl for r in runs] == ["pid"]


def test_a_date_entry_can_name_one_run_or_one_csv():
    """`20260814` = the day, `20260814_2120` = that folder, `20260814_212021`
    = that CSV. The two resolutions exist because the run folder and the CSV
    beside it are stamped differently."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        day = base / "20260814"
        _write_run(day, "2055", "mpc_205520", _square_rows(), _meta("mpc"))
        _write_run(day, "2120", "mpc_212021", _square_rows(), _meta("pid"))
        _write_run(day, "2120", "mpc_212455", _square_rows(), _meta("dobmpc"))

        cfg = _cfg(base, dates=["20260814"])
        assert len(PR.discover(base, _specs(cfg, base), cfg)[0]) == 3

        cfg = _cfg(base, dates=["20260814_2120"])            # the folder
        runs, skips = PR.discover(base, _specs(cfg, base), cfg)
        assert sorted(r.ctrl for r in runs) == ["dobmpc", "pid"] and not skips

        cfg = _cfg(base, dates=["20260814_212021"])          # the one CSV
        runs, skips = PR.discover(base, _specs(cfg, base), cfg)
        assert [r.ctrl for r in runs] == ["pid"] and not skips

        # listing both must not draw the same run twice
        cfg = _cfg(base, dates=["20260814", "20260814_212021"])
        assert len(PR.discover(base, _specs(cfg, base), cfg)[0]) == 3

        cfg = _cfg(base, dates=["20260814_999999"])          # nothing matches
        runs, skips = PR.discover(base, _specs(cfg, base), cfg)
        assert not runs and "20260814_999999" in skips[0]


def test_the_figure_name_carries_the_controller_not_the_csv_prefix():
    """Every controller's log is written as mpc_<hhmmss>.csv — naming the
    figure after the stem would ship a PID run as `..._mpc_....png`."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _write_run(base / "20260814", "2120", "mpc_212021",
                   _square_rows(), _meta("pid"))
        runs, _ = PR.discover(base, [("20260814", "")], _cfg(base))
        assert runs[0].slug == "20260814_2120_pid_212021"
        assert runs[0].csv_path.stem == "mpc_212021"


def test_only_one_error_number_and_which_one():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _write_run(base / "20260814", "2055", "mpc_205520",
                   _square_rows(offset=0.03), _meta())
        r = PR.discover(base, [("20260814", "")], _cfg(base))[0][0]
        # path_following is on in the fixture, so `auto` must NOT quote radial
        # — the reference is held path_lead_m ahead of the vehicle on purpose.
        assert r.headline("auto")[0] == "cross-track RMS"
        assert r.headline("radial")[0] == "radial RMS"
        assert r.headline("radial")[1] > r.headline("cross")[1]


# ------------------------------------------------------------------ the draw
def test_reference_is_drawn_one_lap_only():
    """Five dashed copies at five phases render as a solid line — the drawn
    reference must be a single lap even though the stats use every sample."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _write_run(base / "20260814", "2055", "mpc_205520",
                   _square_rows(laps=3), _meta())
        runs, _ = PR.discover(base, [("20260814", "")], _cfg(base))
        r = runs[0]
        rx, _ry = r.ref_xy(True)
        assert rx.size * 3 == r.rx.size
        assert r.ref_xy(False)[0].size == r.rx.size


def test_figures_are_written_and_share_their_limits():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "sessions"
        day = base / "20260814"
        _write_run(day, "2055", "mpc_205520", _square_rows(), _meta("mpc"))
        _write_run(day, "2120", "mpc_212021", _square_rows(offset=0.06),
                   _meta("pid"))
        cfg = _cfg(base, style={"formats": ["png"], "dpi": 60})
        runs, skips = PR.discover(base, [("20260814", "")], cfg)
        assert len(runs) == 2 and not skips
        box, _ = PR.resolve_pool(cfg, [])
        out = Path(td) / "figs"
        pngs = []
        for r in runs:
            written = PR.plot_run(r, box, cfg, out, plt)
            assert written and written[0].exists() and written[0].stat().st_size
            assert written[0].name == f"{r.slug}.png"
            pngs.append(written[0])
        plt.close("all")

        # Identical geometry is the invariant the whole tool exists for, and
        # the ARTIFACT is where it has to hold: two runs, two PNGs, same size.
        def png_size(p: Path) -> tuple[int, int]:
            b = p.read_bytes()
            i = b.index(b"IHDR") + 4
            return (int.from_bytes(b[i:i + 4], "big"),
                    int.from_bytes(b[i + 4:i + 8], "big"))
        assert png_size(pngs[0]) == png_size(pngs[1])

        # ...and the limits themselves come straight off the box.
        fig, ax = plt.subplots()
        PR._setup_axes(ax, box, cfg["pool"]["pad_m"], PR.THEMES["light"],
                       cfg["style"], cfg)
        pad = float(cfg["pool"]["pad_m"])
        assert ax.get_xlim() == (box["y"][0] - pad, box["y"][1] + pad)
        assert ax.get_ylim() == (box["x"][0] - pad, box["x"][1] + pad)
        plt.close("all")


# --------------------------------------------------------------- the object
#: The 2026-08-21+ schema: the ten obj_* / follow_* columns after e_cross.
OBJ_HEADER = HEADER.rstrip("\n") + (",obj_px,obj_py,obj_pz,obj_yaw_deg,"
                                    "obj_age_s,obj_pair_dt_ms,obj_pair_exact,"
                                    "obj_state,follow_state,follow_err_m\n")


def _obj_rows(base_rows, states, exact, dx=0.6):
    """Bolt an object track onto vehicle rows: the object sits `dx` ahead in
    the datum frame, so its map position is a rigid shift of the vehicle's."""
    out = []
    for i, r in enumerate(base_rows):
        px, py = float(r.split(",")[1]), float(r.split(",")[2])
        st = states[i % len(states)]
        ex = exact[i % len(exact)]
        out.append(r.rstrip("\n") + f",{px + dx:.5f},{py:.5f},0.00000,0.000,"
                                     f"0.100,5.0,{ex},{st},following,0.100\n")
    return out


def test_object_track_is_lifted_with_the_vehicle():
    """obj_px/py are datum-frame world FLU like px/py, so the object rides the
    SAME transform — a fixed offset in the CSV must stay a fixed offset (and
    the same distance) in the map."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        rows = _obj_rows(_square_rows(laps=1), ["live"], ["1"], dx=0.6)
        d = base / "20260814" / "2055"
        d.mkdir(parents=True)
        (d / "mpc_205520.csv").write_text(OBJ_HEADER + "".join(rows))
        (d / "mpc_205520.meta.json").write_text(json.dumps(_meta()))
        runs, _ = PR.discover(base, [("20260814", "")], _cfg(base))
        r = runs[0]
        assert r.has_object and r.ox.size == r.oy.size == r.ot.size
        # every object sample is 0.60 m from the vehicle sample it came from
        i = np.searchsorted(r.t, r.ot)
        i = np.clip(i, 0, r.x.size - 1)
        dist = np.hypot(r.ox - r.x[i], r.oy - r.y[i])
        assert abs(float(np.median(dist)) - 0.6) < 0.02


def test_object_filters_drop_the_rows_the_csv_calls_boundaries():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        # half the rows not live, and of the live ones half are loose pairs
        rows = _obj_rows(_square_rows(laps=1), ["live", "stale"], ["1", "1", "0", "0"])
        d = base / "20260814" / "2055"
        d.mkdir(parents=True)
        (d / "mpc_205520.csv").write_text(OBJ_HEADER + "".join(rows))
        (d / "mpc_205520.meta.json").write_text(json.dumps(_meta()))

        strict = _cfg(base, object={"min_step_m": 0.0})
        loose = _cfg(base, object={"min_step_m": 0.0, "live_only": False,
                                   "pair_exact_only": False})
        n_strict = PR.discover(base, [("20260814", "")], strict)[0][0].ox.size
        n_loose = PR.discover(base, [("20260814", "")], loose)[0][0].ox.size
        assert 0 < n_strict < n_loose
        assert abs(n_strict / n_loose - 0.25) < 0.05      # live AND exact

        off = _cfg(base, object={"show": "off"})
        assert not PR.discover(base, [("20260814", "")], off)[0][0].has_object


def test_a_run_that_never_acquired_says_so_rather_than_showing_nothing():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        rows = _obj_rows(_square_rows(laps=1), ["lost"], ["1"])
        d = base / "20260814" / "2055"
        d.mkdir(parents=True)
        (d / "mpc_205520.csv").write_text(OBJ_HEADER + "".join(rows))
        (d / "mpc_205520.meta.json").write_text(json.dumps(_meta()))
        r = PR.discover(base, [("20260814", "")], _cfg(base))[0][0]
        assert not r.has_object
        assert "never acquired" in r.stats["obj_note"] and "lost" in r.stats["obj_note"]


def test_a_run_without_the_object_columns_still_loads():
    """Every run before 2026-08-21 has no obj_* columns at all."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _write_run(base / "20260814", "2055", "mpc_205520",
                   _square_rows(), _meta())
        r = PR.discover(base, [("20260814", "")], _cfg(base))[0][0]
        assert not r.has_object and not r.stats["obj_note"]


def test_points_outside_the_pool_box_are_counted_not_hidden():
    box = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    assert PR.outside_box(box, [0.5, 0.5], [0.5, 0.5]) == 0
    assert PR.outside_box(box, [0.5, 9.0], [0.5, 0.5]) == 1
    assert PR.outside_box(box, [], []) == 0


# --------------------------------------------------------------- the rest
def test_segment_auto_falls_through_to_the_engagement():
    """A station hold / object follow never sets traj_on, so `traj` would skip
    it. `auto` takes the trajectory when there is one and the engagement when
    there is not."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        day = base / "20260814"
        _write_run(day, "2055", "mpc_205520", _square_rows(), _meta())
        _write_run(day, "2131", "mpc_213102",
                   _square_rows(laps=1, traj_on=False), _meta(traj=False))
        auto = PR.discover(base, [("20260814", "")], _cfg(base))[0]
        assert {r.stats["seg"] for r in auto} == {"trajectory", "engaged"}
        only = PR.discover(base, [("20260814", "")],
                           _cfg(base, segment="traj"))[0]
        assert [r.run_dir.name for r in only] == ["2055"]


def test_a_mid_run_controller_change_is_named_in_the_title():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        rows = _square_rows(laps=2)
        half = len(rows) // 2
        rows = [r.replace(",mpc,1,", ",pid,1,") for r in rows[:half]] + rows[half:]
        d = base / "20260814" / "2055"
        d.mkdir(parents=True)
        (d / "mpc_205520.csv").write_text(HEADER + "".join(rows))
        (d / "mpc_205520.meta.json").write_text(json.dumps(_meta("mpc")))
        r = PR.discover(base, [("20260814", "")], _cfg(base))[0][0]
        assert r.modes == ("pid", "mpc")
        assert r.ctrl_label == "PID  →  MPC"


def test_the_stamp_survives_the_feed_recording_naming():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        d = base / "20260823" / "0823_195927"
        d.mkdir(parents=True)
        stem = "c3_depth_20260823_195927_mpc"
        (d / f"{stem}.csv").write_text(HEADER + "".join(_square_rows(laps=1)))
        (d / f"{stem}.meta.json").write_text(json.dumps(_meta("mpc_tuned")))
        r = PR.discover(base, [("20260823", "")], _cfg(base))[0][0]
        assert r.stamp == "195927", r.stamp        # not the 8-digit date
        assert r.slug == "20260823_0823_195927_mpc_tuned"


def test_a_station_hold_is_not_quoted_as_cross_track():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        meta = _meta(traj=False)
        meta["mission"] = {"config_square": {"shape": "station", "speed": 0.05},
                           "panel_override": {}}
        _write_run(base / "20260814", "2131", "mpc_213102",
                   _square_rows(laps=1, traj_on=False), meta)
        r = PR.discover(base, [("20260814", "")], _cfg(base))[0][0]
        assert r.stats["shape"] == "station"
        assert r.headline("auto")[0] == "radial RMS"
        assert PR._traj_line(meta).startswith("STATION")


def test_a_follow_mission_is_named_a_follow():
    meta = _meta(traj=False)
    meta["mission"] = {"config_square": {"shape": "station", "speed": 0.05},
                       "panel_override": {"shape": "follow", "speed": 0.01}}
    meta["object_nav"] = {"enabled": True, "follow": {"kind": "follow",
                                                      "ended": "tag fix lost"}}
    line = PR._traj_line(meta)
    assert line.startswith("FOLLOW") and "tracked object" in line
    assert "0.01 m/s" in line and "tag fix lost" in line
    assert PR._shape(meta) == "follow"


# ------------------------------------------------------------ time colouring
def _oklab_hex(h):
    import numpy as _np
    c = _np.array([int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return PR._oklab(c.reshape(1, 3))[0]


def _dE(a, b):
    return float(np.linalg.norm(_oklab_hex(a) - _oklab_hex(b)) * 100)


def test_the_time_ramps_are_wide_and_evenly_spaced():
    """The operator could not read time off the first ramps (2026-08-23).

    Two measurable properties fix that, so both are asserted rather than
    eyeballed: the ramp must SPAN enough OKLab to tell the ends apart, and its
    quartiles must be equal so a minute looks like a minute wherever it falls
    — the old orange ramp spent its first half nearly still.
    """
    from matplotlib.colors import to_hex
    for theme in ("light", "dark"):
        for key in ("ramp", "ramp2"):
            cm = PR._cmap(PR.THEMES[theme][key], f"{theme}{key}")
            q = [to_hex(cm(float(v))) for v in (0.0, .25, .5, .75, 1.0)]
            steps = [_dE(q[i], q[i + 1]) for i in range(4)]
            assert _dE(q[0], q[-1]) > 45, (theme, key, _dE(q[0], q[-1]))
            assert max(steps) - min(steps) < 2.5, (theme, key, steps)
            assert min(steps) > 10, (theme, key, steps)


def test_the_ramp_ends_stay_visible_against_their_surface():
    """A trajectory is a thin LINE, so the pale end may not recede into the
    surface the way a heat-map cell is allowed to."""
    from matplotlib.colors import to_hex

    def lum(h):
        c = np.array([int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])

    for theme in ("light", "dark"):
        th = PR.THEMES[theme]
        surf = lum(th["surface"])
        for key in ("ramp", "ramp2"):
            cm = PR._cmap(th[key], f"c{theme}{key}")
            end = to_hex(cm(0.0 if theme == "light" else 1.0))
            a, b = max(lum(end), surf), min(lum(end), surf)
            assert (a + 0.05) / (b + 0.05) >= 2.0, (theme, key, end)


def test_banding_flattens_the_ramp_into_readable_steps():
    from matplotlib.colors import to_hex
    cm = PR._cmap(PR.THEMES["light"]["ramp"], "b", bands=4)
    shades = {to_hex(cm(float(v))) for v in np.linspace(0, 1, 40)}
    assert len(shades) == 4, shades


def test_time_ticks_land_on_round_seconds():
    t = np.arange(0.0, 500.0, 0.05)
    marks = PR._tick_times(t)
    assert marks.size and all(float(m) % 60 == 0 for m in marks), marks
    assert marks[0] == 60 and marks[-1] <= t[-1]
    # a short run gets a proportionally short interval, not the same one
    assert PR._tick_times(np.arange(0.0, 8.0, 0.05))[0] == 1
    assert PR._tick_times(np.array([0.0])).size == 0


def test_a_tick_with_no_sample_near_it_is_dropped_not_moved():
    """`segment: engaged` leaves gaps where the run was released. Snapping the
    120 s tick onto the nearest sample at 141 s would put a wrong time on a
    right-looking place, so that mark gets no tick at all."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 0-100 s flown, 100-200 s released, 200-300 s flown again
    t = np.concatenate([np.arange(0.0, 100.0, 0.05),
                        np.arange(200.0, 300.0, 0.05)])
    ang = 2 * math.pi * t / 60.0
    r = PR.Run(date="20260814", run_dir=Path("2055"),
               csv_path=Path("mpc_205520.csv"), meta=_meta())
    r.x, r.y, r.t = np.cos(ang), np.sin(ang), t

    marks = PR._tick_times(t)
    step = float(marks[1] - marks[0])
    i = np.clip(np.searchsorted(t, marks), 0, t.size - 1)
    kept = np.abs(t[i] - marks) <= max(1.0, 0.05 * step)
    assert 0 < kept.sum() < marks.size, (kept.sum(), marks.size)
    # the dropped ones are exactly the marks inside the hole
    assert all(100.0 < float(m) < 200.0 for m in marks[~kept]), marks[~kept]

    fig, ax = plt.subplots()
    PR._setup_axes(ax, {"x": [-2, 2], "y": [-2, 2]}, 0.05,
                   PR.THEMES["light"], PR.DEFAULTS["style"], PR.DEFAULTS)
    PR._draw_time_ticks(ax, r, PR.THEMES["light"], PR.DEFAULTS["style"],
                        0, 0.0, float(t[-1]))
    drawn = {float(tx.get_text()) for tx in ax.texts}
    assert drawn, "no tick labels drawn"
    assert all(v % step == 0 for v in drawn), drawn
    assert not any(100.0 < v < 200.0 for v in drawn), drawn
    plt.close("all")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
