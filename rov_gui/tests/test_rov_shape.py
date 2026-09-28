#!/usr/bin/env python3
"""
test_rov_shape.py — the vehicle's drawn geometry, offline and without Qt.

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_rov_shape.py

`widgets/rov_shape.py` is the body-frame model the trajectory panel draws:
hull, 8 thrusters, the Newton gripper, the C3. Almost all of it is
arithmetic nobody looks at, in a frame conversion this repo has already got
wrong once in production (hw_nav.yaml's `cam_t_flu` block: a C3 extrinsic
with the lens 0.155 m BELOW the COM when it is 0.105 m above, which threw
the vehicle 0.46 m into the air in this very view and was cited as device
fact for weeks). So what is pinned here is:

* BOTH sign flips of body FLU -> map NED, against hand-computed vectors
  rather than against the module's own arithmetic;
* that the drawn footprint is corner-for-corner the rectangle the panel drew
  before the gripper existed — the marker must not move by a millimetre;
* that the drawn TCP is the SAME point the controller composes plans at
  (hw_mpc.yaml `policy.tcp_body_flu_m`), because two copies of a TCP is how
  a picture ends up disagreeing with what it illustrates;
* that the vertical chain closes: the jaw sits 47.075 mm above the underside
  and the hull is 0.254 m tall, which is the cross-check hw_nav.yaml used to
  prove the old C3 z impossible — and that the jaw is anchored to the LENS
  (JAW_CENTRE_M == CAM_T_FLU_M + LENS_TO_GRIP_FLU_M, the 2026-09-08
  measurement), not to that chain's refuted horizontal 110.664 mm;
* that every constant has a provenance entry and every entry a tag — the
  CLAUDE.md rule, enforced rather than trusted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.widgets import rov_shape as RS

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ frames
def test_body_to_map_flips_y_and_z_and_nothing_else():
    """Hand-computed, not derived from the module. At yaw 0: forward stays
    +x; LEFT becomes -y (map y is EAST); UP becomes -z (map z is DOWN)."""
    o = (0.0, 0.0, 0.0)
    assert RS.body_to_map((1.0, 0.0, 0.0), o, 0.0) == (1.0, 0.0, 0.0)
    assert RS.body_to_map((0.0, 1.0, 0.0), o, 0.0) == (0.0, -1.0, 0.0)
    assert RS.body_to_map((0.0, 0.0, 1.0), o, 0.0) == (0.0, 0.0, -1.0)


def test_body_to_map_rotates_about_the_map_z_and_translates():
    """At yaw +90 deg (heading EAST) forward must point along map +y, and
    the body origin must land exactly on the position given."""
    p = RS.body_to_map((1.0, 0.0, 0.0), (5.0, -2.0, 0.3), math.pi / 2.0)
    assert abs(p[0] - 5.0) < 1e-12 and abs(p[1] - (-2.0 + 1.0)) < 1e-12
    assert abs(p[2] - 0.3) < 1e-12
    o = RS.body_to_map((0.0, 0.0, 0.0), (5.0, -2.0, 0.3), 1.234)
    assert max(abs(a - b) for a, b in zip(o, (5.0, -2.0, 0.3))) < 1e-12


def test_body_to_map_preserves_length():
    """A rotation plus two reflections is still an isometry. A scale error
    here would shrink or stretch the whole vehicle silently."""
    v = (0.31, -0.22, 0.17)
    n0 = math.sqrt(sum(c * c for c in v))
    for yaw in (0.0, 0.7, -1.9, math.pi, 2.6):
        m = RS.body_to_map(v, (0.0, 0.0, 0.0), yaw)
        assert abs(math.sqrt(sum(c * c for c in m)) - n0) < 1e-12


# --------------------------------------------------------------- footprint
def test_footprint_is_exactly_the_rectangle_the_panel_always_drew():
    """`TrajectoryView._hull`, transcribed. The vehicle marker gaining a
    gripper must not move the rectangle the operator has been judging wall
    clearance against since 2026-08-14."""
    def old_hull(p, yaw, L, W):
        hl, hw = L / 2.0, W / 2.0
        ca, sa = math.cos(yaw), math.sin(yaw)
        return [(p[0] + ca * dx - sa * dy, p[1] + sa * dx + ca * dy)
                for dx, dy in ((hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw))]

    for yaw in (0.0, 0.7, -1.9, math.pi, 2.6):
        for p in ((0.0, 0.0), (1.3, -2.2)):
            a = old_hull(p, yaw, RS.HULL_L_M, RS.HULL_W_M)
            b = RS.footprint(p, yaw)
            assert len(a) == len(b) == 4
            for (ax, ay), (bx, by) in zip(a, b):
                assert abs(ax - bx) < 1e-12 and abs(ay - by) < 1e-12


# ---------------------------------------------------------------- the TCP
def test_drawn_tcp_is_the_controllers_tcp():
    """The ring on the plot and `policy.tcp_body_flu_m` must be ONE point."""
    import yaml

    with open(REPO / "config" / "hw_mpc.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    want = [float(v) for v in cfg["policy"]["tcp_body_flu_m"]]
    got = [float(v) for v in RS.tcp_body_m()]
    assert max(abs(a - b) for a, b in zip(want, got)) < 1e-9, (want, got)


def test_drawn_camera_is_the_navs_camera():
    """Same rule for the C3: the ray must start where the localizer's
    extrinsic says the lens is, and point along the angle it applies."""
    import yaml

    with open(REPO / "config" / "hw_nav.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    want = [float(v) for v in cfg["cam_t_flu"]]
    got = [float(v) for v in RS.CAM_T_FLU_M]
    assert max(abs(a - b) for a, b in zip(want, got)) < 1e-9, (want, got)
    assert abs(float(cfg["cam_tilt_deg"]) - RS.CAM_TILT_DEG) < 1e-9
    # ...and the sign that was wrong in production: the lens is ABOVE.
    assert RS.CAM_T_FLU_M[2] > 0.0, "the C3 sign error is back"
    ax = RS.camera_axis_flu()
    assert ax[0] > 0.0 and ax[2] < 0.0, "the C3 looks forward and DOWN"


# ------------------------------------------------------- the vertical chain
def test_the_vertical_chain_closes_on_hw_nav_s_numbers():
    """hw_nav.yaml's `cam_t_flu` block is the only well-sourced statement of
    this vehicle's vertical extent, and it is a CHAIN: the jaw sits
    47.075 mm above the underside, the lens 275.014 mm above the jaw, and
    the whole vehicle is 0.254 m tall. If the drawn hull stops agreeing with
    it, the picture and the localizer are describing different vehicles.
    The chain's HORIZONTAL link is a different story — see below."""
    jaw_above_underside = RS.JAW_CENTRE_M[2] - RS.UNDERSIDE_Z_M
    assert abs(jaw_above_underside - 0.047075) < 1e-9
    lens_above_jaw = RS.CAM_T_FLU_M[2] - RS.JAW_CENTRE_M[2]
    assert abs(lens_above_jaw - 0.275014) < 1e-5, lens_above_jaw
    # The HORIZONTAL half of that CAD chain — "lens 110.664 mm back of the
    # jaw-pair centre" — is REFUTED: with the recorded PnP pose that jaw
    # projects to row ~420 of the 360-row colour frame while the jaws are
    # visible at rows 265-360, and a jar held in the jaws and set on the
    # centre of floor tag 58 puts the grip point 0.196 m ahead of the lens
    # (range 0.187-0.204) [측정 2026-09-08: data/20260908/
    # 0908_180453/policy_obs/rgb/000160.jpg (tag PnP, the run's own
    # intrinsics) + 0908_170428/policy_obs/rgb/000000.jpg (jar height
    # 0.084 m) + 0908_175151/policy_obs/rgb/000000.jpg (open jaws)]. So the
    # jaw is anchored to the LENS, and the literal JAW_CENTRE_M must be that
    # sum to within rounding (2 mm) — whichever half of the chain is wrong.
    assert abs(RS.LENS_TO_GRIP_FLU_M[0] - 0.196) < 1e-9, RS.LENS_TO_GRIP_FLU_M
    assert abs(RS.LENS_TO_GRIP_FLU_M[2] + 0.275) < 1e-9, "the CAD vertical"
    for k in range(3):
        want = RS.CAM_T_FLU_M[k] + RS.LENS_TO_GRIP_FLU_M[k]
        assert abs(RS.JAW_CENTRE_M[k] - want) < 2e-3, (k, RS.JAW_CENTRE_M, want)
    lens_back_of_jaw = RS.JAW_CENTRE_M[0] - RS.CAM_T_FLU_M[0]
    assert lens_back_of_jaw > 0.15, "the refuted CAD 0.1107 is back"
    # the COM sits 0.217 m above the underside — hw_nav's own [유도]
    assert abs(-RS.UNDERSIDE_Z_M - 0.217) < 1e-3
    top = RS.UNDERSIDE_Z_M + RS.HULL_H_M
    assert abs(RS.HULL_Z_M - (RS.UNDERSIDE_Z_M + top) / 2.0) < 1e-12
    # ...and the impossibility argument hw_nav used still holds: the old,
    # wrong lens z would put the COM taller than the whole vehicle.
    assert 0.477 > RS.HULL_H_M


def test_the_gripper_is_tucked_above_the_underside_not_below_it():
    """It is mounted on the bottom panel BETWEEN the skids, so from above
    only its overhang past the nose is visible — which is what
    `TrajectoryView._draw_rov_flat` clips to. If the tube ever sits below
    the hull, that clip becomes a lie."""
    tube_bottom = RS.GRIP_CENTRE_M[2] - RS.GRIP_R_M
    assert tube_bottom > RS.UNDERSIDE_Z_M, (tube_bottom, RS.UNDERSIDE_Z_M)
    nose = RS.HULL_L_M / 2.0
    assert RS.JAW_CENTRE_M[0] - RS.JAW_HALF_M[0] > nose, "jaws are hidden"
    assert RS.GRIP_CENTRE_M[0] - RS.GRIP_L_M / 2.0 < nose, "tube fully out"


# ----------------------------------------------------------------- parts
def test_every_part_is_a_closed_box_with_outward_normals():
    """Eight vertices, six quads, and a normal that points AWAY from the
    part's centre. A flipped normal is invisible until the 3-D shading is
    inside out and the far faces are the ones drawn."""
    for part in RS.parts():
        v, faces = part["verts"], part["faces"]
        assert len(v) == 8, part["name"]
        assert len(faces) == 6, part["name"]
        cx = tuple(sum(q[k] for q in v) / 8.0 for k in range(3))
        for idx, n in faces:
            assert len(idx) == 4 and len(set(idx)) == 4
            fc = tuple(sum(v[i][k] for i in idx) / 4.0 for k in range(3))
            out = tuple(fc[k] - cx[k] for k in range(3))
            assert sum(out[k] * n[k] for k in range(3)) > 0, (part["name"], n)
            assert abs(math.sqrt(sum(c * c for c in n)) - 1.0) < 1e-9


def test_the_model_has_the_parts_the_operator_asked_to_see():
    names = {p["name"] for p in RS.parts()}
    assert {"hull", "tube", "jaw_left", "jaw_right", "camera"} <= names
    assert len([n for n in names if n.startswith("thruster_")]) == 8


def test_jaws_are_mirrored_and_open_outward():
    """The pair straddles the centreline and both jaws move AWAY from it as
    the opening grows — a sign slip here closes the jaws when told to open."""
    closed = {p["name"]: p for p in RS.parts(jaw_open_frac=0.0)}
    wide = {p["name"]: p for p in RS.parts(jaw_open_frac=1.0)}
    for tag in ("jaw_left", "jaw_right"):
        yc = sum(v[1] for v in closed[tag]["verts"]) / 8.0
        yw = sum(v[1] for v in wide[tag]["verts"]) / 8.0
        assert abs(yw) > abs(yc), tag
        assert yc * yw > 0.0, f"{tag} crossed the centreline"
    yl = sum(v[1] for v in wide["jaw_left"]["verts"]) / 8.0
    yr = sum(v[1] for v in wide["jaw_right"]["verts"]) / 8.0
    assert abs(yl + yr) < 1e-12, "the pair is not symmetric"
    # ...and the sim's own clear gap, which is NOT the vendor's 0 -> 62 mm.
    gap_closed = 2.0 * (RS.JAW_CLOSED_HALF_GAP_M - RS.JAW_HALF_M[1])
    gap_open = 2.0 * (RS.JAW_CLOSED_HALF_GAP_M + RS.JAW_TRAVEL_M
                      - RS.JAW_HALF_M[1])
    assert abs(gap_closed - 0.010) < 1e-9 and abs(gap_open - 0.072) < 1e-9
    assert "10 mm clear gap" in RS.GEOMETRY_PROVENANCE["JAW_TRAVEL_M"]


def test_thruster_shrouds_span_the_drawn_footprint_width():
    """The shroud radius is DERIVED from the footprint on the assumption
    that the vertical thrusters are what makes the vehicle that wide. If the
    two ever stop closing, the derivation has lost its only evidence."""
    outer = max(abs(c[1]) for c, _a in RS.THRUSTERS_M) + RS.THRUSTER_R_M
    assert abs(2.0 * outer - RS.HULL_W_M) < 1e-6, outer


def test_thruster_axes_are_unit_and_four_are_vectored():
    horiz = [a for _c, a in RS.THRUSTERS_M if abs(a[2]) < 0.5]
    vert = [a for _c, a in RS.THRUSTERS_M if abs(a[2]) >= 0.5]
    assert len(horiz) == 4 and len(vert) == 4
    for a in horiz:
        assert abs(abs(a[0]) - abs(a[1])) < 1e-9, "not a 45 deg vector"
    for _c, a in RS.THRUSTERS_M:
        assert abs(math.sqrt(sum(v * v for v in a)) - 1.0) < 1e-9


def test_silhouette_of_a_rotated_thruster_is_a_diamond_not_its_box():
    """The bug this function exists to fix: a 45-deg thruster's bounding box
    is 41 % wider than the shroud, and drawing it hid which way it points."""
    t0 = next(p for p in RS.parts() if p["name"] == "thruster_0")
    hull2d = RS.silhouette_xy(t0["verts"])
    assert len(hull2d) >= 4
    xs = [q[0] for q in hull2d]
    ys = [q[1] for q in hull2d]
    box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    area = 0.5 * abs(sum(hull2d[i][0] * hull2d[(i + 1) % len(hull2d)][1]
                         - hull2d[(i + 1) % len(hull2d)][0] * hull2d[i][1]
                         for i in range(len(hull2d))))
    assert area < 0.85 * box_area, (area, box_area)


# ----------------------------------------------------------- heading line
def test_the_heading_line_runs_through_the_com_both_ways():
    """Operator request 2026-09-07: the only line on the vehicle used to be
    the optical-axis ray, which starts at the lens and stops ahead of the
    gripper. The heading line is the vehicle's x-axis THROUGH the COM: it
    must reach further ahead than the ray ever did, and behind the hull's
    tail — and lie in the vehicle's horizontal plane, so in 3-D it is not
    the ray."""
    aft, fwd = RS.heading_line_body()
    # on the axis, at the COM's height
    assert aft[1] == 0.0 and aft[2] == 0.0 and fwd[1] == 0.0 and fwd[2] == 0.0
    assert aft[0] < 0.0 < fwd[0]
    # longer than the ray's top-down reach (lens x + ray length * cos tilt)
    ax = RS.camera_axis_flu()
    ray_reach = RS.CAM_T_FLU_M[0] + RS.CAM_RAY_M * ax[0]
    assert fwd[0] > ray_reach + 0.1, (fwd[0], ray_reach)
    # ...and past the tail, not just to it
    assert -aft[0] > RS.HULL_L_M / 2.0 + 0.1, (aft[0], RS.HULL_L_M)
    # ...but not so long the plan's whole 3 s horizon hides under it
    assert fwd[0] <= 1.5 and -aft[0] <= 1.5
    # the ray still pitches down; the heading line does not
    assert ax[2] < 0.0


def test_the_heading_line_follows_the_yaw_through_body_to_map():
    """At yaw +90 deg (EAST) the forward end must sit along map +y and
    the aft end along map -y — the same flip-free rotation the hull uses,
    so the line and the body can never point different ways."""
    aft, fwd = RS.heading_line_body()
    o = (2.0, -1.0, 0.3)
    a = RS.body_to_map(aft, o, math.pi / 2.0)
    b = RS.body_to_map(fwd, o, math.pi / 2.0)
    assert abs(a[0] - 2.0) < 1e-9 and abs(b[0] - 2.0) < 1e-9
    assert abs(a[1] - (-1.0 + aft[0])) < 1e-9, a
    assert abs(b[1] - (-1.0 + fwd[0])) < 1e-9, b
    assert abs(a[2] - 0.3) < 1e-9 and abs(b[2] - 0.3) < 1e-9


# ------------------------------------------------------ recorded-run TCP
def test_tcp_for_a_recorded_run_follows_its_meta_not_the_live_module():
    """The offline tools draw a RECORDED run's TCP dot from that run's meta:
    plans.jsonl p_tcp / anchor_tcp and the meta's T_body_tcp were recorded
    with the TCP the station carried that day, and the jaw moved on
    2026-09-08 (0.4165 -> 0.502 m ahead of the COM). A dot from the live
    module on a 2026-09-07 run would sit 8.5 cm ahead of the run's own plan
    polyline — two record generations in one picture."""
    old = [0.4165, 0.0, -0.17]
    v, src = RS.tcp_body_for_meta({"rov_drawn_geometry": {"jaw_centre_m": old},
                                   "policy": {"tcp": {"tcp_body_flu_m": old}}})
    assert v == (0.4165, 0.0, -0.17) and "rov_drawn_geometry" in src, src
    # a run that predates rov_drawn_geometry but recorded the policy TCP
    v, src = RS.tcp_body_for_meta({"policy": {"tcp": {"tcp_body_flu_m": old}}})
    assert v == (0.4165, 0.0, -0.17) and "policy.tcp" in src, src
    # nothing recorded (or no meta at all): the live module, and it SAYS so
    for meta in ({}, None, {"policy": {}}, {"rov_drawn_geometry": {}},
                 {"rov_drawn_geometry": {"jaw_centre_m": [1.0, 2.0]}}):
        v, src = RS.tcp_body_for_meta(meta)
        assert v == tuple(RS.tcp_body_m()) and "live" in src, (meta, src)
    # today's meta round-trips to today's jaw
    v, _ = RS.tcp_body_for_meta(RS.rov_geometry_meta())
    assert v == tuple(RS.JAW_CENTRE_M)


# ------------------------------------------------------------- provenance
def test_every_constant_has_a_tagged_provenance_entry():
    """CLAUDE.md, enforced: a number in code presented as measured carries
    its artifact path, and anything else carries [예측]/[유도]/[스펙]."""
    named = [n for n in dir(RS)
             if n.isupper() and not n.startswith("_")
             and n != "GEOMETRY_PROVENANCE"]
    for n in named:
        assert n in RS.GEOMETRY_PROVENANCE, f"{n} has no provenance"
    for k, v in RS.GEOMETRY_PROVENANCE.items():
        assert any(t in v for t in ("[측정", "[스펙", "[유도", "[예측")), k


def test_the_meta_carries_the_whole_table():
    m = RS.rov_geometry_meta()
    assert "FLU" in m["frame"] and "COM" in m["frame"]
    assert m["provenance"] == RS.GEOMETRY_PROVENANCE
    assert m["jaw_opening_is_live"] is False, \
        "the station has no jaw feedback; the drawn opening is schematic"


# =============================================================================
# runner (same shape as the other suites — works with or without pytest)
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
