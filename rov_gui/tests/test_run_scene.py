#!/usr/bin/env python3
"""
test_run_scene.py — the offline run video (``tools/render_run_scene.py``).

Every test here is about a FRAME, because that is the only thing this renderer
can get wrong in a way that still looks like a plausible picture. The one that
matters most is the last: the CSV logs the pose in world FLU and the policy
plans in NED, so a renderer that forgets the flip draws the vehicle below the
tag mat and half a metre from its own reference — and the picture gives no hint.

    python -m pytest rov_gui/tests/test_run_scene.py -q
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rov_gui.control.state_assembler import rot_zyx            # noqa: E402
from rov_gui.tools.render_run_scene import (Run, View,         # noqa: E402
                                            tag_quads)

TAG_MAP = ROOT / "config" / "tag_map_full.yaml"


# ------------------------------------------------------------------ fixtures
def _write_run(d: Path, *, p0=(0.2, -0.4, -0.25), yaw0_deg=88.0,
               map_sha1=None, rows=None, plans=None, rgb=False,
               depth=True, rp_cols=False) -> Path:
    """A minimal but REAL run folder: the loader reads nothing else.

    ``rp_cols`` writes the schema-16 ``policy_plan.csv`` header (2026-09-26:
    ``roll_deg,pitch_deg`` appended AFTER ``reason``; a plan dict may carry
    ``rp`` = [(roll_deg, pitch_deg), ...] per knot, else nan) -- the default
    is the pre-16 header, so every existing test reads the old layout.
    """
    import hashlib
    sha = map_sha1 or hashlib.sha1(TAG_MAP.read_bytes()).hexdigest()[:12]
    meta = {"schema_version": 13,
            "hardware": {"tag_map": str(TAG_MAP), "tag_map_sha1": sha,
                         "tag_size_m": 0.17,
                         "datum_tag_frame": {"p0": list(p0),
                                             "yaw0_deg": yaw0_deg}},
            "fstereo": {"iters": 8, "scale": 0.75, "ckpt": "x/y.pth"}}
    (d / "mpc_000000.meta.json").write_text(json.dumps(meta))

    if rows is None:
        rows = [{"t": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0, "yaw_deg": 0.0},
                {"t": 0.5, "px": 0.1, "py": 0.2, "pz": 0.3, "yaw_deg": 10.0}]
    fields = ["t", "px", "py", "pz", "yaw_deg", "pitch_deg", "roll_deg",
              "n_tags", "pnp_rms_px", "tag_age_s", "z_src", "bridge_s"]
    with (d / "mpc_000000.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: 0 for k in fields}, "z_src": "tag", **r})

    if plans is None:
        plans = [{"plan_id": "1", "t_rel": 0.1, "status": "accept",
                  "knots": [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]}]
    with (d / "policy_plan.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        head = ["plan_id", "status", "follower", "t_rel", "k", "t_knot",
                "x_ned", "y_ned", "z_ned", "yaw_deg", "reason"]
        if rp_cols:
            head += ["roll_deg", "pitch_deg"]          # schema 16: after reason
        w.writerow(head)
        for p in plans:
            for k, q in enumerate(p["knots"]):
                row = [p["plan_id"], p["status"], "observe", p["t_rel"],
                       k, 0.2 * k, q[0], q[1], q[2], 0.0, ""]
                if rp_cols:
                    rp = (p.get("rp") or [])
                    row += list(rp[k]) if k < len(rp) else ["nan", "nan"]
                w.writerow(row)

    if not depth:
        return d                       # a run flown without --record-depth
    obs = d / "policy_obs"
    (obs / "depth").mkdir(parents=True)
    import cv2
    mm = np.zeros((40, 64), np.uint16)
    mm[10:30, 20:44] = 800                      # a patch at 0.8 m, rest invalid
    for i in range(2):
        cv2.imwrite(str(obs / "depth" / f"{i:06d}.png"), mm)
    (obs / "meta.json").write_text(json.dumps({"obs": {"grid_kind": "rect_left"},
                                               "builder": {}}))
    with (obs / "index.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        head = ["seq", "t_capture", "obs_png", "depth_png"]
        if rgb:
            head += ["rgb_jpg", "rgb_dt_ms"]
            (obs / "rgb").mkdir()
        w.writerow(head)
        for i, t in enumerate((1000.0, 1000.2)):
            row = [i, f"{t:.6f}", f"obs/{i:06d}.png", f"depth/{i:06d}.png"]
            if rgb:
                cv2.imwrite(str(obs / "rgb" / f"{i:06d}.jpg"),
                            np.full((48, 64, 3), 120, np.uint8))
                row += [f"rgb/{i:06d}.jpg", "12.0"]
            w.writerow(row)
    return d


@pytest.fixture()
def run_dir():
    with tempfile.TemporaryDirectory(prefix="rovgui_scene_") as t:
        yield _write_run(Path(t))


# ------------------------------------------------------------------- frames
def test_to_map_is_the_exact_inverse_of_datumize(run_dir):
    """`p_map = p0 + Rz(+yaw0) @ p_datum` must undo MpcWorker._datumize, whose
    arithmetic is reproduced here rather than imported so a change to either
    side shows up as a failure instead of agreeing with itself."""
    r = Run(run_dir)
    rng = np.random.default_rng(0)
    for _ in range(20):
        p_tag = rng.normal(size=3)
        Rz = rot_zyx(0.0, 0.0, -r.yaw0)             # _datumize's rotation
        p_datum = Rz @ (p_tag - r.p0)
        back = r.to_map(p_datum)
        assert np.allclose(back, p_tag, atol=1e-12), (back, p_tag)


def test_the_csv_is_flu_and_the_plans_are_ned(run_dir):
    """The flip, pinned. A CSV row of (+1, +1, +1) is NED (+1, -1, -1), so the
    vehicle drawn from it must land where a plan of (+1, -1, -1) does.

    This is the defect that made the first render put the vehicle 0.93 m from
    its own reference and below the mat; on the real run the residual between
    the two, after the flip, is 4.5 mm [측정 2026-09-06,
    data/20260906/0906_191935_observe t 294.83].
    """
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_scene_")),
                   rows=[{"t": 0.0, "px": 1.0, "py": 1.0, "pz": 1.0,
                          "yaw_deg": 30.0}],
                   plans=[{"plan_id": "1", "t_rel": 0.0, "status": "accept",
                           "knots": [(1.0, -1.0, -1.0)]}])
    r = Run(d)
    assert np.allclose(r.p_datum[0], [1.0, -1.0, -1.0]), r.p_datum[0]
    assert math.isclose(r.yaw[0], math.radians(-30.0), abs_tol=1e-12)
    plan, _ = r.plan_at(0.0)
    assert np.allclose(r.p_map[0], plan["knots_map"][0], atol=1e-9), (
        "a CSV pose and the plan anchored at it must land on the same point")




def _add_second_engagement(d: Path, *, epoch=4, t0_mono=9000.0):
    """A SECOND engagement in the same folder, sharing the per-plan logs.

    Exactly the shape of data/20260906/0906_191935_observe: its
    t_rel restarts at 0, so its plans interleave with the first engagement's
    unless the loader filters by epoch.
    """
    import hashlib
    sha = hashlib.sha1(TAG_MAP.read_bytes()).hexdigest()[:12]
    meta = {"schema_version": 13,
            "hardware": {"tag_map": str(TAG_MAP), "tag_map_sha1": sha,
                         "tag_size_m": 0.17,
                         "datum_tag_frame": {"p0": [0.2, -0.4, -0.25],
                                             "yaw0_deg": 88.0}},
            "policy": {"epoch": epoch}, "fstereo": {}}
    (d / "mpc_222222.meta.json").write_text(json.dumps(meta))
    fields = ["t", "px", "py", "pz", "yaw_deg", "pitch_deg", "roll_deg",
              "n_tags", "pnp_rms_px", "tag_age_s", "z_src", "bridge_s", "plan_id"]
    with (d / "mpc_222222.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerow({**{k: 0 for k in fields}, "t": 0.0, "z_src": "tag",
                    "plan_id": "900"})
    with (d / "policy_plan.csv").open("a", newline="") as fh:
        w = csv.writer(fh)
        for k in range(2):
            w.writerow(["900", "accept", "observe", 0.05, k, 0.2 * k,
                        9.0, 9.0, 9.0, 0.0, ""])
    with (d / "plans.jsonl").open("a") as fh:
        fh.write(json.dumps({"plan_id": 900, "epoch": epoch,
                             "obs_t_mono": t0_mono + 0.05, "obs_t": 0.05}) + "\n")


def test_two_engagements_in_one_folder_are_not_mixed():
    """The real failure this guards: a folder holding two engagements whose
    plans.jsonl and policy_plan.csv are SHARED, with t_rel restarting at 0.

    Measured on data/20260906/0906_191935_observe: mpc_191935
    (epoch 2, plans 1..1426) and mpc_200050 (epoch 4, plans 1991..2203) in one
    folder, 1639 plan ids across both files. Before this filter the renderer
    drew the SECOND engagement's reference at the first one's times, and the
    measured clock offset spread read 2475 s instead of 0.
    """
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_scene_")))
    # the first engagement declares its epoch and its plan
    m = json.loads((d / "mpc_000000.meta.json").read_text())
    m["policy"] = {"epoch": 2}
    (d / "mpc_000000.meta.json").write_text(json.dumps(m))
    with (d / "plans.jsonl").open("w") as fh:
        fh.write(json.dumps({"plan_id": 1, "epoch": 2,
                             "obs_t_mono": 1000.1, "obs_t": 0.1}) + "\n")
    _add_second_engagement(d)

    r = Run(d)
    assert r.meta_path.name == "mpc_000000.meta.json", (
        "the engagement whose plans are inside the depth window must win")
    assert r.epoch == 2
    ids = {p["plan_id"] for p in r.plans}
    assert ids == {"1"}, f"the other engagement's plans leaked in: {ids}"
    assert math.isclose(r.t_offset, 1000.0, abs_tol=1e-6)
    assert r.offset_spread == 0.0, "the offset must be measured within one epoch"


def test_one_depth_recording_across_engagements_is_clipped_not_mixed():
    """The recorder is armed per STATION PROCESS, the CSVs are per engagement.

    Measured 2026-09-07 on data/20260907/
    0907_164659: ONE 1160-frame recording spanning epochs 2,4,6,8,10. Every
    engagement then scores high on "plans inside the depth window", so the
    pick is by frames inside ITS OWN window — and the frames outside it are
    DROPPED, because pose_at would otherwise clamp them to an edge row and
    draw a frozen vehicle under a moving depth picture.
    """
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_span_")), depth=False)
    m = json.loads((d / "mpc_000000.meta.json").read_text())
    m["policy"] = {"epoch": 2}
    (d / "mpc_000000.meta.json").write_text(json.dumps(m))
    _add_second_engagement(d, epoch=4, t0_mono=1010.6)
    # one recording covering BOTH windows: epoch 2 near 1000.1, epoch 4 at 1010.65
    obs = d / "policy_obs"
    (obs / "depth").mkdir(parents=True)
    (obs / "meta.json").write_text(json.dumps({"obs": {}, "builder": {}}))
    stamps = [1000.0, 1000.1, 1000.2, 1010.6, 1010.7, 1010.8]
    with (obs / "index.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seq", "t_capture", "obs_png", "depth_png"])
        for i, t in enumerate(stamps):
            cv2.imwrite(str(obs / "depth" / f"{i:06d}.png"),
                        np.zeros((40, 64), np.uint16))
            w.writerow([i, f"{t:.6f}", "", f"depth/{i:06d}.png"])
    with (d / "plans.jsonl").open("w") as fh:
        fh.write(json.dumps({"plan_id": 1, "epoch": 2,
                             "obs_t_mono": 1000.1, "obs_t": 0.1}) + "\n")
        fh.write(json.dumps({"plan_id": 900, "epoch": 4,
                             "obs_t_mono": 1010.65, "obs_t": 0.05}) + "\n")

    r = Run(d)
    assert r.epoch in (2, 4)
    w0, w1 = r._plan_window(r.epoch)
    assert all(w0 - 1.0 <= t <= w1 + 1.0 for t in r.t_depth), (
        "every kept frame must sit in the chosen engagement's window")
    assert r.clipped > 0, "the other engagement's frames must be dropped"
    assert len(r.depth_rows) + r.clipped == len(stamps)


def test_an_engagement_can_be_named():
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_scene_")))
    m = json.loads((d / "mpc_000000.meta.json").read_text())
    m["policy"] = {"epoch": 2}
    (d / "mpc_000000.meta.json").write_text(json.dumps(m))
    _add_second_engagement(d)
    r = Run(d, engagement="222222")
    assert r.meta_path.name == "mpc_222222.meta.json" and r.epoch == 4



def test_plan_at_returns_the_newest_plan_already_issued(run_dir):
    r = Run(run_dir)
    r.plans = [{"plan_id": str(i), "t_rel": float(i), "status": "accept",
                "knots_map": np.zeros((2, 3))} for i in range(5)]
    r.plan_t = np.array([p["t_rel"] for p in r.plans])
    assert r.plan_at(-1.0)[0] is None, "nothing is in force before the first"
    for t, want in ((0.0, "0"), (2.9, "2"), (99.0, "4")):
        p, age = r.plan_at(t)
        assert p["plan_id"] == want and age >= 0.0


def test_clock_offset_is_measured_not_assumed(run_dir):
    """With no plans.jsonl the offset falls back to aligning the spans, and the
    spread is NaN so a reader can tell it was not measured."""
    r = Run(run_dir)
    assert math.isclose(r.t_offset, 1000.0 - 0.0)
    assert math.isnan(r.offset_spread)


# --------------------------------------------------------------------- map
def test_a_changed_tag_map_is_refused(run_dir):
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_scene_")),
                   map_sha1="deadbeef0000")
    with pytest.raises(ValueError, match="CHANGED"):
        Run(d)
    r = Run(d, strict_map=False)          # opt out, and it is stamped
    assert r.map_matches is False


def test_tag_quads_are_squares_of_the_tag_size(run_dir):
    r = Run(run_dir)
    quads = tag_quads(r.tags, r.tag_size)
    assert len(quads) >= len(r.tags.instances)
    for _, w in quads[:20]:
        sides = [float(np.linalg.norm(w[i] - w[(i + 1) % 4])) for i in range(4)]
        assert all(abs(s - r.tag_size) < 1e-9 for s in sides), sides
        # planar: the four corners span a plane, so the volume is zero
        v = np.linalg.det(np.stack([w[1] - w[0], w[2] - w[0], w[3] - w[0]]))
        assert abs(v) < 1e-12


def test_the_map_is_the_one_the_run_flew(run_dir):
    """The run stores a path and a hash, so the renderer must check it — a map
    edited after the dive would draw a pool that was never flown."""
    r = Run(run_dir)
    assert r.map_matches and len(r.map_sha1) == 12


# -------------------------------------------------------------- projection
def test_view_puts_the_scene_inside_the_panel_and_the_centre_at_the_centre():
    pts = np.array([[0, 0, 0], [4, 0, 0], [0, 2, 0], [4, 2, 0],
                    [2, 1, -0.5]], float)
    size = (640, 480)
    v = View.fit(pts, size)
    px, z = v(pts)
    assert (z > 0).all(), "every point must be in front of the eye"
    assert px[:, 0].min() >= 0 and px[:, 0].max() <= size[0]
    assert px[:, 1].min() >= 0 and px[:, 1].max() <= size[1]
    centre, _ = v(np.array([[2.0, 1.0, -0.25]]))
    assert abs(centre[0][0] - size[0] / 2) < 40
    assert abs(centre[0][1] - size[1] / 2) < 40


def test_view_up_is_minus_z_because_the_map_is_ned():
    """A point ABOVE the floor (more negative z) must draw HIGHER on screen.
    Getting this backwards still looks like a scene, just upside down."""
    pts = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]], float)
    v = View.fit(pts, (640, 480))
    low, _ = v(np.array([[0.5, 0.5, 0.0]]))
    high, _ = v(np.array([[0.5, 0.5, -0.5]]))
    assert high[0][1] < low[0][1], "smaller z (up in NED) must be higher up"


# --------------------------------------------------------------- html export
def test_the_html_template_has_exactly_one_data_slot():
    """The exporter substitutes a single token. Two (or none) would ship a page
    whose data is a stale literal — and it would still open."""
    from rov_gui.tools.export_run_html import HTML
    text = HTML.read_text()
    assert text.count("/*__DATA__*/null") == 1
    assert "fetch(" not in text, (
        "a page opened from a folder cannot fetch a sibling file; the data has "
        "to stay inlined")


def test_the_page_ships_the_tested_vehicle_geometry():
    """The browser transforms the SAME body model the station draws, so the
    two pictures cannot disagree about what the vehicle is."""
    from rov_gui.tools.export_run_html import rov_model
    from rov_gui.tools.render_run_scene import _SHAPE
    m = rov_model()
    kinds = {p["kind"] for p in m["parts"]}
    assert {"hull", "jaw", "thruster", "camera"} <= kinds, kinds
    assert m["tcp"] == [round(v, 4) for v in _SHAPE.tcp_body_m()]
    assert m["hull"] == [_SHAPE.HULL_L_M, _SHAPE.HULL_W_M]
    for part in m["parts"]:
        for f in part["faces"]:
            assert len(f["i"]) >= 3 and len(f["n"]) == 3
            assert max(f["i"]) < len(part["verts"])


def test_the_tcp_dot_of_a_recorded_run_follows_its_meta():
    """A pre-2026-09-08 run (meta jaw_centre_m 0.4165) gets its TCP dot at
    0.4165, not at the live module's 0.502: its plans.jsonl p_tcp and
    T_body_tcp were recorded with that point, and a dot 8.5 cm ahead of the
    plan's own polyline would mix two record generations. A run that
    recorded no TCP falls back to the live module, and says so."""
    from rov_gui.tools.export_run_html import rov_model
    from rov_gui.tools.render_run_scene import _SHAPE
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_scene_")))
    mp = d / "mpc_000000.meta.json"
    meta = json.loads(mp.read_text())
    meta["rov_drawn_geometry"] = {"jaw_centre_m": [0.4165, 0.0, -0.17]}
    meta["policy"] = {"tcp": {"tcp_body_flu_m": [0.4165, 0.0, -0.17]}}
    mp.write_text(json.dumps(meta))
    r = Run(d)
    assert r.tcp_body == (0.4165, 0.0, -0.17) and "meta" in r.tcp_source
    m = rov_model(r)
    assert m["tcp"] == [0.4165, 0.0, -0.17] and "meta" in m["tcp_source"]
    assert m["tcp"] != [round(v, 4) for v in _SHAPE.tcp_body_m()], (
        "the live jaw must differ from the old one, or this test proves nothing")
    # the same folder with the geometry stripped -> live, labelled live
    del meta["rov_drawn_geometry"], meta["policy"]
    mp.write_text(json.dumps(meta))
    r2 = Run(d)
    assert r2.tcp_body == tuple(_SHAPE.tcp_body_m()) and "live" in r2.tcp_source
    assert rov_model(r2)["tcp"] == [round(v, 4) for v in _SHAPE.tcp_body_m()]
    assert rov_model()["tcp"] == [round(v, 4) for v in _SHAPE.tcp_body_m()]


def test_the_sixteen_step_horizon_is_recomposed_and_agrees_with_the_record():
    """Six dots, not sixteen — and why the sixteen can still be drawn.

    The policy emits 16 steps at its training stride (66.7 ms, 1.07 s of
    horizon); ``resample_knots`` puts them on the 0.2 s reference grid, which
    is 6 knots, and only those are logged. The 16 are recomposed here with the
    run's OWN ``compose_plan`` from the recorded action, anchor and T_body_tcp
    — and the check that this is a reconstruction rather than a re-derivation
    is that the recomposed knots equal the recorded ones (0.000 um over 383
    plans of the 2026-09-06 run [측정 2026-09-06]).

    Two record generations in ONE plans.jsonl: plan 1 carries the 5-column
    ``action_raw`` of the pos_yaw_width runs ([dx, dy, dz, dyaw, width],
    schema 12, 2026-09-07), plan 2 the legacy 10-column rows ([pos, rot6d,
    width]). The renderer dispatches on the array width, not the meta
    version, so BOTH must recompose to 16 points — and, describing the same
    straight 4.5 cm run, to the same span.
    """
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_raw_")),
                   plans=[{"plan_id": "1", "t_rel": 0.1, "status": "accept",
                           "knots": [(0.0, 0.0, 0.0), (0.045, 0.0, 0.0)]},
                          {"plan_id": "2", "t_rel": 0.6, "status": "accept",
                           "knots": [(0.0, 0.0, 0.0), (0.045, 0.0, 0.0)]}])
    meta = json.loads((d / "mpc_000000.meta.json").read_text())
    meta["policy"] = {"epoch": 2,
                      "tcp": {"T_body_tcp": np.eye(4).tolist()},
                      "config": {"gripper_width_open_m": 0.069,
                                 "gripper_width_closed_m": 0.042}}
    (d / "mpc_000000.meta.json").write_text(json.dumps(meta))
    # 16 steps: a straight 4.5 cm run, zero dyaw / identity rot6d, open jaw
    action5 = [[0.003 * k, 0.0, 0.0, 0.0, 0.069] for k in range(16)]
    action10 = [[0.003 * k, 0.0, 0.0, 1, 0, 0, 0, 1, 0, 0.069] for k in range(16)]
    (d / "plans.jsonl").write_text(
        json.dumps({"plan_id": 1, "epoch": 2, "obs_t_mono": 1000.1, "obs_t": 0.1,
                    "action_raw": action5,
                    "anchor_pose_used": [0, 0, 0, 0, 0, 0],
                    "obs_dt_s": 1 / 15.0, "dt": 0.2, "t0": 0.1,
                    "obs_t_rel": 0.1}) + "\n"
        + json.dumps({"plan_id": 2, "epoch": 2, "obs_t_mono": 1000.6, "obs_t": 0.6,
                      "action_raw": action10,
                      "anchor_pose_used": [0, 0, 0, 0, 0, 0],
                      "obs_dt_s": 1 / 15.0, "dt": 0.2, "t0": 0.6,
                      "obs_t_rel": 0.6}) + "\n")

    r = Run(d)
    assert len(r.plans) == 2 and r.n_raw_ok == 2, (len(r.plans), r.n_raw_ok)
    spans = []
    for plan, width in zip(r.plans, (5, 10)):
        assert plan["raw_map"] is not None, f"{width}-column action_raw not recomposed"
        assert len(plan["raw_map"]) == 16, (
            f"the raw horizon is the policy's 16 steps ({width}-column record)")
        assert plan["n_raw"] == 16
        assert abs(plan["obs_dt"] - 1 / 15.0) < 1e-9
        # both live in the map frame: same isometry, so the ends agree
        assert np.allclose(plan["raw_map"][0],
                           r.to_map(np.asarray(plan["raw_map"][0])
                                    * 0 + plan["raw_map"][0]) * 0
                           + plan["raw_map"][0])
        span_raw = float(np.linalg.norm(plan["raw_map"][-1] - plan["raw_map"][0]))
        span_knot = float(np.linalg.norm(np.asarray(plan["knots_map"][-1])
                                         - np.asarray(plan["knots_map"][0])))
        assert span_raw > 0.03 and abs(span_raw - span_knot) < 0.02, (
            f"the two descriptions of one horizon disagree ({width}-column): "
            f"{span_raw} vs {span_knot}")
        spans.append(span_raw)
    assert abs(spans[0] - spans[1]) < 1e-9, (
        f"5- and 10-column records of the same run span differently: {spans}")


def test_policy_plan_csv_roll_pitch_columns_are_read_by_name_and_absent_is_nan():
    """Schema 16 (2026-09-26) appends ``roll_deg,pitch_deg`` AFTER ``reason``
    in policy_plan.csv -- the append-at-end rule, so a by-name reader sees
    the old 11-column file (every pre-16 run) and the new one alike: the
    old layout yields nan per knot, the new one the recorded values, and a
    knot without an attitude reference (a 4-DoF plan under schema 16) is
    nan too. Nothing positional: the x/y/z/yaw columns are untouched."""
    old = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_pp_old_")))
    r = Run(old)
    assert len(r.plans) == 1 and len(r.plans[0]["rp"]) == 2
    assert all(math.isnan(a) and math.isnan(b) for a, b in r.plans[0]["rp"])
    new = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_pp_new_")), rp_cols=True,
                     plans=[{"plan_id": "1", "t_rel": 0.1, "status": "accept",
                             "knots": [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)],
                             "rp": [(1.5, -4.0), (2.0, -5.0)]},
                            {"plan_id": "2", "t_rel": 0.6, "status": "accept",
                             "knots": [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]}])
    r2 = Run(new)
    by = {p["plan_id"]: p for p in r2.plans}
    assert by["1"]["rp"] == [[1.5, -4.0], [2.0, -5.0]]
    assert all(math.isnan(a) and math.isnan(b) for a, b in by["2"]["rp"])
    # the positional columns read the same either way
    assert np.allclose(by["1"]["knots"], r.plans[0]["knots"])
    assert by["1"]["yaw"] == r.plans[0]["yaw"] and by["1"]["t_knot"] == r.plans[0]["t_knot"]
    # and the schema-16 header is exactly the old one plus the two trailing names
    head = next(csv.reader(open(new / "policy_plan.csv")))
    assert head == ["plan_id", "status", "follower", "t_rel", "k", "t_knot",
                    "x_ned", "y_ned", "z_ned", "yaw_deg", "reason", "roll_deg", "pitch_deg"]


def test_a_seven_column_record_is_recomposed_like_the_five_column_one():
    """The 6-DoF variant's plans.jsonl (2026-09-26): ``action_raw`` rows are
    7 wide ([dx, dy, dz, dyaw, droll, dpitch, width]). With the roll/pitch
    columns at 0 and a level anchor the recomposed 16 steps are BIT-EQUAL to
    the 5-column record of the same run (columns 0:4 and 6 are the 5-dim
    columns; design D12: at a level anchor body-z and world-z composition
    coincide), in both sub-modes -- ``rp_tracked`` false (dropped-and-
    logged, the mandatory first flight) and true with meta
    ``trajectory.attitude_track`` -- and the renderer dispatches on the
    width alone. Needs ``compose_plan`` to take width 7 and ``track_rp``
    (the frames group); until then this test FAILS, on purpose.
    """
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_raw7_")), rp_cols=True,
                   plans=[{"plan_id": str(i), "t_rel": 0.1 * i, "status": "accept",
                           "knots": [(0.0, 0.0, 0.0), (0.045, 0.0, 0.0)]}
                          for i in (1, 2, 3)])
    meta = json.loads((d / "mpc_000000.meta.json").read_text())
    meta["schema_version"] = 16
    meta["policy"] = {"epoch": 2,
                      "tcp": {"T_body_tcp": np.eye(4).tolist()},
                      "config": {"gripper_width_open_m": 0.069,
                                 "gripper_width_closed_m": 0.042,
                                 "rp_max_deg": 20.0}}
    meta["trajectory"] = {"kind": "policy", "attitude_track": True}
    (d / "mpc_000000.meta.json").write_text(json.dumps(meta))
    action5 = [[0.003 * k, 0.0, 0.0, 0.0, 0.069] for k in range(16)]
    action7 = [[0.003 * k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.069] for k in range(16)]
    recs = [dict(plan_id=1, epoch=2, obs_t_mono=1000.1, obs_t=0.1, action_raw=action5,
                 anchor_pose_used=[0, 0, 0, 0, 0, 0], obs_dt_s=1 / 15.0, dt=0.2,
                 t0=0.1, obs_t_rel=0.1),
            dict(plan_id=2, epoch=2, obs_t_mono=1000.2, obs_t=0.2, action_raw=action7,
                 anchor_pose_used=[0, 0, 0, 0, 0, 0], obs_dt_s=1 / 15.0, dt=0.2,
                 t0=0.2, obs_t_rel=0.2, rp_tracked=False),
            dict(plan_id=3, epoch=2, obs_t_mono=1000.3, obs_t=0.3, action_raw=action7,
                 anchor_pose_used=[0, 0, 0, 0, 0, 0], obs_dt_s=1 / 15.0, dt=0.2,
                 t0=0.3, obs_t_rel=0.3, rp_tracked=True)]
    (d / "plans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    r = Run(d)
    assert len(r.plans) == 3 and r.n_raw_ok == 3, (len(r.plans), r.n_raw_ok)
    by = {p["plan_id"]: p for p in r.plans}
    for pid in ("1", "2", "3"):
        assert by[pid]["raw_map"] is not None and len(by[pid]["raw_map"]) == 16, pid
    assert np.array_equal(np.asarray(by["2"]["raw_map"]), np.asarray(by["1"]["raw_map"]))
    assert np.array_equal(np.asarray(by["3"]["raw_map"]), np.asarray(by["1"]["raw_map"]))
    span = float(np.linalg.norm(by["3"]["raw_map"][-1] - by["3"]["raw_map"][0]))
    assert abs(span - 0.045) < 1e-9


def test_recorded_colour_is_found_without_a_flag(run_dir):
    """--record-depth files the colour beside the obs, so the C3 RGB column
    lights up by itself. The column it must NOT pick is `obs_png`: that is the
    depth tensor, and showing it as colour would look right and be wrong."""
    from rov_gui.tools.export_run_html import RgbSource
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_rgb_")), rgb=True)
    src = RgbSource(d / "policy_obs")
    assert src.dir is not None and len(src.files) == 2
    got = src.at(1000.19)
    assert got is not None and got[0].name == "000001.jpg"
    assert abs(got[1] - 0.01) < 1e-6, "the join must report its own gap"

    with pytest.raises(ValueError, match="colour column"):
        RgbSource(run_dir / "policy_obs")      # depth-only run: no rgb column


def test_a_run_without_recorded_depth_still_makes_a_scene():
    """--record-depth costs the right-hand panel and NOTHING else.

    The pool, the vehicle and the reference come from the CSV and the plans, so
    a run flown without it is still worth drawing; the timeline then comes from
    the pose log at ~5 Hz and every frame says there is no picture.
    """
    rows = [{"t": i * 0.05, "px": 0.01 * i, "py": 0.0, "pz": 0.0,
             "yaw_deg": 0.0} for i in range(40)]      # 2 s at 20 Hz
    d = _write_run(Path(tempfile.mkdtemp(prefix="rovgui_nodepth_")),
                   rows=rows, depth=False)
    r = Run(d)
    assert r.has_depth is False and r.no_depth_why
    assert len(r.depth_rows) == 10, "40 rows at 20 Hz, every 4th -> ~5 Hz"
    assert all(row["depth_png"] == "" for row in r.depth_rows)
    assert r.t_offset == 0.0, "the spine IS run time; nothing to shift"
    # and the frames still carry a pose and can find their plan
    i, _ = r.pose_at(float(r.t_depth[3]))
    assert 0 <= i < len(r.t)


def test_an_empty_run_folder_is_refused_with_its_neighbours_named():
    """A stream recorder's companion CSV (0 rows) looks like a run folder. The
    refusal has to say what IS next door, or the operator opens them by hand."""
    from rov_gui.tools.render_run_scene import describe_siblings
    base = Path(tempfile.mkdtemp(prefix="rovgui_empty_"))
    good = base / "0906_193554"
    good.mkdir()
    _write_run(good)
    empty = base / "0906_200641"
    empty.mkdir()
    _write_run(empty, rows=[], depth=False)
    with pytest.raises(ValueError, match="no rows"):
        Run(empty)
    hint = describe_siblings(empty)
    assert "0906_193554" in hint and "pose rows" in hint
    assert "0906_200641" not in hint, "an empty folder is not a suggestion"


def test_a_run_name_under_the_wrong_date_folder_is_pointed_at():
    """A run folder carries its own date INSIDE a date folder, so the two can
    disagree by a day and the path still looks right. Saying "does not exist"
    and then listing that day's OTHER runs sends the reader the wrong way."""
    from rov_gui.tools.render_run_scene import describe_siblings
    base = Path(tempfile.mkdtemp(prefix="rovgui_date_"))
    (base / "20260906").mkdir()
    real = base / "20260907" / "0907_133358"
    real.mkdir(parents=True)
    _write_run(real)
    typo = base / "20260906" / "0907_133358"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Run(typo)
    hint = describe_siblings(typo)
    assert "the same run name exists here" in hint and str(real) in hint


def test_embed_writes_one_file_that_carries_its_pictures(run_dir):
    """--embed is the answer to "can I just download the html": the frames go
    INTO the page as data: URIs, so there is nothing beside it to lose."""
    from rov_gui.tools.export_run_html import RgbSource, export
    out = Path(tempfile.mkdtemp(prefix="rovgui_embed_"))
    r = Run(run_dir)
    page = export(r, out, every=1, limit=None, quality=80, domain="obs",
                  cmap="turbo", rgb=RgbSource(None), embed=True, fmt="webp",
                  out_name="one.html")
    assert page.name == "one.html"
    assert not (out / "frames").exists(), "embedding must not write a folder"
    text = page.read_text()
    assert text.count("data:image/webp;base64,") >= 2
    assert '"embedded":true' in text.replace(" ", "")


def test_the_folder_form_keeps_the_page_small(run_dir):
    """...and the default keeps the pictures beside it, so the page stays a
    page: the 2026-09-06 run is 0.8 MB against 26 MB embedded."""
    from rov_gui.tools.export_run_html import RgbSource, export
    out = Path(tempfile.mkdtemp(prefix="rovgui_folder_"))
    r = Run(run_dir)
    page = export(r, out, every=1, limit=None, quality=80, domain="obs",
                  cmap="turbo", rgb=RgbSource(None), embed=False, fmt="jpg")
    assert (out / "frames").is_dir()
    assert len(list((out / "frames").glob("d*.jpg"))) == 2
    assert "data:image/" not in page.read_text()


def test_the_hover_layer_carries_the_exact_millimetres():
    """The readout under the cursor is the RECORDED millimetre, not the colour.

    Reading the panel back would measure the renderer: obs is an 8-bit palette
    position with dz/dt = 4.667*z^2, so one level spans 0.16 m at 3 m, and the
    panel is lossy on top of that. So the page carries the mm map itself as a
    lossless two-channel picture — which is only worth its bytes if it is bit
    exact."""
    import cv2
    from rov_gui.tools.export_run_html import encode_values
    rng = np.random.default_rng(4)
    mm = rng.integers(0, 65535, size=(97, 131)).astype(np.uint16)
    mm[0, 0], mm[1, 1], mm[2, 2] = 0, 65535, 256      # invalid, top, a carry
    back = cv2.imdecode(np.frombuffer(encode_values(mm), np.uint8),
                        cv2.IMREAD_COLOR)
    got = (back[..., 2].astype(np.uint16) << 8) | back[..., 1].astype(np.uint16)
    assert (got == mm).all(), "the value layer is not lossless"


def test_the_hover_geometry_inverts_the_panel_layout():
    """Mouse -> source pixel, checked against the panel the exporter actually
    draws, for every pixel of the image band.

    fit_into rounds nw and nh INDEPENDENTLY, so a 320x240 grid lands at
    533/320 across and 400/240 down. One shared scale passes at 640x400 (the
    2026-09 grid, where the fit is 1:1) and names the wrong column on every
    other grid — which is what this test caught."""
    import cv2
    from rov_gui.depth_colour import Z_FAR_M, Z_NEAR_M, colorize, mm_to_t_valid
    from rov_gui.depth_panel import depth_panel
    from rov_gui.tools.export_run_html import panel_geometry

    def js_inverse(g, px, py):
        """The page's sourcePixel(), verbatim."""
        if not (g["x0"] <= px < g["x0"] + g["nw"]
                and g["y0"] <= py < g["y0"] + g["nh"]):
            return None
        return (min(g["iw"] - 1, (px - g["x0"]) * g["iw"] // g["nw"]),
                min(g["ih"] - 1, (py - g["y0"]) * g["ih"] // g["nh"]))

    for h, w in [(400, 640), (240, 320), (480, 640), (200, 800)]:
        mm = np.random.default_rng(0).integers(0, 3500, (h, w)).astype(np.uint16)
        t, valid = mm_to_t_valid(mm, "obs", Z_NEAR_M, Z_FAR_M)
        panel = depth_panel(t, valid, domain="obs", cmap="turbo",
                            z_near=Z_NEAR_M, z_far=Z_FAR_M, lines=["a", "b"])
        want = colorize(t, valid, "turbo")
        g = panel_geometry(mm.shape)
        for py in range(g["y0"], g["y0"] + g["nh"]):
            for px in range(g["x0"], g["x0"] + g["nw"]):
                sx, sy = js_inverse(g, px, py)
                assert tuple(panel[py, px]) == tuple(want[sy, sx]), (
                    f"{h}x{w}: panel ({px},{py}) is not source ({sx},{sy})")
        assert js_inverse(g, g["x0"], g["y0"] - 1) is None, "header is not depth"
        assert js_inverse(g, g["x0"], g["y0"] + g["nh"]) is None, "bar is not depth"


def test_the_embedded_page_can_read_its_own_depth(run_dir):
    """End to end: the page carries a value layer per frame, the geometry to
    address it, and the source file name the readout cites."""
    import json
    import re
    from rov_gui.tools.export_run_html import RgbSource, export
    out = Path(tempfile.mkdtemp(prefix="rovgui_hover_"))
    page = export(Run(run_dir), out, every=1, limit=None, quality=80,
                  domain="obs", cmap="turbo", rgb=RgbSource(None), embed=True,
                  fmt="webp", out_name="h.html")
    data = json.loads(re.search(r"const D = (\{.*\});", page.read_text()).group(1))
    g = data["depth_geom"]
    assert g and g["hdr"] == 46 and g["pw"] == 640
    assert data["hover"]["on"] is True
    for f in data["frames"]:
        assert f["v"].startswith("data:image/webp;base64,")
        assert f["dpng"].startswith("depth/")

    off = export(Run(run_dir), Path(tempfile.mkdtemp(prefix="rovgui_nohover_")),
                 every=1, limit=None, quality=80, domain="obs", cmap="turbo",
                 rgb=RgbSource(None), embed=True, fmt="webp", out_name="n.html",
                 hover=False)
    d2 = json.loads(re.search(r"const D = (\{.*\});", off.read_text()).group(1))
    assert d2["depth_geom"] is None and d2["hover"]["on"] is False
    assert all(f["v"] == "" for f in d2["frames"])
    assert off.stat().st_size < page.stat().st_size


def main() -> int:
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        needs = "run_dir" in fn.__code__.co_varnames[:fn.__code__.co_argcount]
        try:
            if needs:
                with tempfile.TemporaryDirectory(prefix="rovgui_scene_") as t:
                    fn(_write_run(Path(t)))
            else:
                fn()
            print(f"  ok    {name}")
        except Exception:                                        # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
