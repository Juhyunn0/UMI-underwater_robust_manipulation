#!/usr/bin/env python3
"""
rov_shape.py — the vehicle's own geometry, as a body-frame model the
trajectory panel can draw in both of its modes.

Until 2026-09-03 the panel drew the vehicle as ONE oriented rectangle: its
footprint, centred on the position the tag solution reports. That is the
vehicle's COM, and on this vehicle the thing the operator actually cares
about is 0.50 m in front of it — the JAW. The diffusion policy's TCP *is*
the jaw (`hw_mpc.yaml policy.tcp_body_flu_m`), so every reference the DP
draws is composed there, and a marker centred on the COM put the vehicle's
picture half a metre from the point its trajectory was about.

So this module holds the parts, in metres, in the vehicle BODY frame, and
`parts()` returns them as convex polygons ready to project. No Qt here on
purpose: the frame algebra is the part that can be wrong in a way nobody
notices, and it is unit-tested without a display (tests/test_rov_shape.py).

    ┌─────────────┐
    │             │──────┬═╡   +x (forward, heading)
    │    hull     │ tube  jaws  <- TCP = jaw-pair centre
    │             │──────┴═╡
    └─────────────┘

FRAMES — the one thing in this file that must not be got wrong.

    BODY is FLU:  +x forward, +y LEFT,  +z UP     (origin = vehicle COM)
    MAP  is NED:  +x north,   +y east,  +z DOWN

so converting body to map is TWO sign flips (y and z) plus the yaw
rotation, and `body_to_map` is the only place either happens. This repo has
already shipped one sign error of exactly this kind — hw_nav.yaml's
`cam_t_flu` block documents a C3 extrinsic that had the lens 0.155 m BELOW
the COM when it is 0.105 m above, and it was cited as device fact for weeks
before anyone caught it. Hence `test_rov_shape.py` pins both flips against
hand-computed values rather than against this file's own arithmetic.

PROVENANCE (CLAUDE.md: a number in code presented as measured carries its
artifact path; otherwise it is tagged). Every dimension below is in
`GEOMETRY_PROVENANCE`, and `rov_geometry_meta()` hands the whole table to
the run meta so a recorded run says which picture it was flown with.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# the vehicle, in body FLU metres (origin = vehicle COM)
# ---------------------------------------------------------------------------
#: Hull footprint (along-heading, across). The panel's existing
#: `rov_size_m`, and the window overrides both from hw_nav.yaml
#: `rov_footprint_m` — kept here as the fallback so this module stands alone.
HULL_L_M = 0.4318
HULL_W_M = 0.5334

#: The vehicle's UNDERSIDE, and from it the hull box's height and centre.
#: Both come from the operator's CAD chain in hw_nav.yaml's `cam_t_flu`
#: block, which is the best-sourced vertical statement in this repo — it is
#: the chain that CAUGHT the 0.26 m C3 sign error, and its own consistency
#: argument ("0.477 m above the underside is taller than the whole vehicle")
#: is what pins the total height.
UNDERSIDE_Z_M = -0.217075   # = JAW_CENTRE z (-0.17) - 47.075 mm
HULL_H_M = 0.254
HULL_Z_M = UNDERSIDE_Z_M + HULL_H_M / 2.0      # -> -0.090075

#: Newton Subsea Gripper tube: a cylinder along +x under the front bottom.
#: NOTE the tube sits 47 mm ABOVE the underside, not below it — the gripper
#: is tucked between the skids, so the only part of it that shows from above
#: is the overhang past the nose. `_draw_rov_flat` clips to exactly that.
#: x = the sim's GRIP_POS 0.25 shifted by the same +0.0855 the jaw moved on
#: 2026-09-08 (LENS_TO_GRIP_FLU_M below), so the tube still ends at the jaws.
GRIP_CENTRE_M = (0.3355, 0.0, -0.17)
GRIP_R_M = 0.018
GRIP_L_M = 0.303

#: Lens -> grip point (the jaw-pair centre, where a ~5 cm object is held),
#: body FLU. THE one gripper dimension MEASURED on this vehicle (2026-09-08):
#: a jar held in the jaws and set on the centre of floor tag 58 put the grip
#: point 0.196 m ahead of the C3 lens — not the 0.1107 m the CAD chain in
#: hw_nav.yaml's `cam_t_flu` block implies (that jaw projects to image row
#: ~420 of the 360-row colour frame, while the jaws are plainly visible at
#: rows 265-360). So the jaw is anchored to the LENS: whichever half of the
#: chain is wrong (the CAD offset, or cam_t_flu's own x/z — open, see
#: KNOWN_ISSUES 2026-09-08), lens + this vector is where the jaws are. z is
#: the CAD vertical, which the same image confirms to ~1 cm.
LENS_TO_GRIP_FLU_M = (0.196, 0.0, -0.275)

#: The jaw pair at its CLOSED position, and how far each jaw travels.
#: `JAW_HALF` is (half-length along x, half-thickness along y, half-height).
#: JAW_CENTRE_M is CAM_T_FLU_M + LENS_TO_GRIP_FLU_M (= 0.50184, 0, -0.16999)
#: written as a rounded LITERAL so it stays byte-identical to hw_mpc.yaml
#: `policy.tcp_body_flu_m`; test_rov_shape pins the sum within 2 mm and the
#: config equality exactly. If CAM_T_FLU_M ever changes, re-derive BOTH.
JAW_CENTRE_M = (0.502, 0.0, -0.17)
JAW_HALF_M = (0.030, 0.003, 0.012)
JAW_CLOSED_HALF_GAP_M = 0.008
JAW_TRAVEL_M = 0.031       # per jaw; 62 mm total opening

#: The 8 T200s, as (centre, thrust axis) in body FLU. Positions are the sim's
#: thruster SITES with the composite-COM offset removed, so they land in the
#: same vehicle-COM frame as everything else here; the axis is the site quat
#: applied to local +X, which is the thrust direction the model and the
#: allocation both use (bluerov_heavy_gripper.xml:30 and thrusters.py:206).
#: 0-3 are the vectored horizontals at +-45 deg, 4-7 the verticals.
_R2 = math.sqrt(0.5)
THRUSTERS_M = (
    ((0.13550, -0.10000, -0.07250), (_R2, _R2, 0.0)),      # 0 fwd-stbd
    ((0.13550, 0.10000, -0.07250), (_R2, -_R2, 0.0)),      # 1 fwd-port
    ((-0.14750, -0.10000, -0.07250), (-_R2, _R2, 0.0)),    # 2 aft-stbd
    ((-0.14750, 0.10000, -0.07250), (-_R2, -_R2, 0.0)),    # 3 aft-port
    ((0.12000, -0.22000, -0.00500), (0.0, 0.0, 1.0)),      # 4 vertical
    ((0.12000, 0.22000, -0.00500), (0.0, 0.0, 1.0)),       # 5
    ((-0.12000, -0.22000, -0.00500), (0.0, 0.0, 1.0)),     # 6
    ((-0.12000, 0.22000, -0.00500), (0.0, 0.0, 1.0)),      # 7
)
#: Shroud radius and length. The radius is DERIVED, not looked up: the
#: verticals sit at y = +-0.220 and the measured footprint is 0.5334 wide, so
#: the shroud that defines that edge has radius 0.5334/2 - 0.220 = 0.0467.
#: That the two independent numbers close to 4 mm is the only cross-check
#: available in this repo — there is no T200 dimension in it.
THRUSTER_R_M = 0.0467
THRUSTER_L_M = 0.075

#: The MarineSitu C3, the stereo camera the diffusion policy SEES through.
#: Drawn because on a DP run the operator's question is "is the bottle in
#: frame", and that is answered by where the lens is and which way it looks —
#: which is not obvious from a hull outline, the lens being 0.105 m ABOVE the
#: COM and pitched 43.3 deg DOWN.
CAM_T_FLU_M = (0.30584, 0.0, 0.10501)
CAM_DIMS_M = (0.095, 0.165, 0.089)      # optical depth, width, height
CAM_TILT_DEG = 43.3                     # DOWN from level
#: How long the optical-axis ray is drawn. A pointer, not a range claim.
CAM_RAY_M = 0.45

#: THE HEADING LINE: a dotted line through the COM along body +x, drawn
#: this far AHEAD of the COM and this far BEHIND it. Operator request
#: 2026-09-07: from above the only line on the vehicle was the optical-axis
#: ray, which starts at the lens (over the gripper tube) and stops 0.63 m
#: ahead of the COM — it read as "the heading line", and a heading line
#: that exists only in front of the gripper is useless for judging whether
#: the plan / the object lies ON the vehicle's axis or beside it. So the
#: axis is drawn through the whole body, both ways. Lengths are a drawing
#: choice: 0.85 m ahead of the COM is ~0.32 m past the jaw tip (0.532 m),
#: about three-quarters of one hull length (0.4318 m); 0.55 m behind is
#: ~0.33 m past the tail (0.2159 m), about the same. Long enough to sight
#: along, short enough not to become a fence at the panel's usual zoom.
#: (The old ray reached 0.306 + 0.45 cos 43.3 deg = 0.633 m ahead [유도].)
HEADING_FWD_M = 0.85
HEADING_AFT_M = 0.55

#: What fraction of its travel the drawn jaw sits at. The station has NO jaw
#: feedback — `PayloadState.gripper_fb` is None on the hardware ROV and the
#: whole jaw path is open-loop — so an animated jaw would be a picture of a
#: position nobody knows. Drawn at a fixed, obviously-schematic opening
#: instead; see the module docstring of widgets/payload.py for the same rule
#: applied to the numeric readout.
JAW_DRAW_OPEN_FRAC = 0.5

GEOMETRY_PROVENANCE = {
    "HULL_L_M": "[측정: operator 2026-08-14, config/hw_nav.yaml "
                "`rov_footprint_m` — 17 x 21 inch, heading along the 17 inch "
                "side] (the window overrides this from that key)",
    "HULL_W_M": "[측정: operator 2026-08-14, config/hw_nav.yaml "
                "`rov_footprint_m`]",
    "UNDERSIDE_Z_M": "[유도: config/hw_nav.yaml `cam_t_flu` block (the "
                     "47.075 + 275.014 = 322.1 mm chain) — the operator's CAD "
                     "chain puts the jaw-pair centre 47.075 mm above the "
                     "ROV's underside, and that centre is at z -0.17, so the "
                     "underside is at -0.217075]",
    "HULL_H_M": "[스펙: config/hw_nav.yaml `cam_t_flu` block — 'the whole vehicle "
                "(0.254 m)', the number that cross-check uses to prove the "
                "old C3 z impossible]",
    "HULL_Z_M": "[유도: UNDERSIDE_Z_M + HULL_H_M/2. NOT the sim's collision "
                "geom, which was the first source tried and is a DYNAMICS "
                "proxy: its 0.50 x 0.35 plan disagrees with the measured "
                "footprint because it excludes the thruster shrouds, and "
                "its z would have put the hull 40 mm off this chain.]",
    "GRIP_CENTRE_M": "[유도: bluerov2_mujoco_marinegym/"
                     "compute_payload_inertia.py:55 GRIP_POS x 0.25 ('cylinder "
                     "centre (bottom panel front)', an ESTIMATE of where the "
                     "gripper was mounted [예측]) shifted by the same +0.0855 "
                     "the jaw moved when it was anchored to the lens on "
                     "2026-09-08, so the tube still ends at the jaws. The "
                     "tube's own position on this vehicle is NOT measured — "
                     "only the jaw is (LENS_TO_GRIP_FLU_M); the vendor gives "
                     "the tube and the opening. The sim's GRIP_POS itself is "
                     "untouched (it is the plant's mass composition).]",
    "GRIP_R_M": "[스펙: compute_payload_inertia.py:56 GRIP_R]",
    "GRIP_L_M": "[스펙: compute_payload_inertia.py:56 GRIP_L]",
    "LENS_TO_GRIP_FLU_M": "[측정 2026-09-08: x from data/*/*_observe/"
                          "20260908/0908_180453/policy_obs/rgb/000160.jpg (jar "
                          "gripped on tag 58; tag PnP with the run's own "
                          "intrinsics) + jar height 0.084 (±0.006) m from "
                          "data/20260908/0908_170428_observe/policy_obs/rgb/"
                          "000000.jpg + open-jaw rays in data/*/*_observe/"
                          "20260908/0908_175151/policy_obs/rgb/000000.jpg; range "
                          "0.187-0.204; z = the CAD 275.014 mm vertical, "
                          "image-consistent to ~1 cm; y = 0 assumed [예측] — the "
                          "colour camera sees the gripped jar 0.011 m RIGHT of "
                          "the body axis in the same frames, not resolved here]",
    "JAW_CENTRE_M": "[유도: CAM_T_FLU_M + LENS_TO_GRIP_FLU_M = (0.50184, 0, "
                    "-0.16999), rounded to (0.502, 0, -0.17); jaw-pair centre "
                    "at the CLOSED position; identical to hw_mpc.yaml "
                    "policy.tcp_body_flu_m, i.e. this IS the policy's TCP. "
                    "Anchored to the LENS on 2026-09-08: the CAD chain's 0.4165 "
                    "(compute_payload_inertia.py:95 JAW_POS, = lens + 110.664 "
                    "mm) is contradicted by the C3 image (measured 196 mm). "
                    "Right under BOTH open attributions (KNOWN_ISSUES "
                    "2026-09-08 'cam_t_flu x/z'): if cam_t_flu is right the "
                    "sim's jaw sits 8.5 cm too far aft; if cam_t_flu is ~9 cm "
                    "too far forward the sim's jaw was about right. RECORD "
                    "BOUNDARY: runs before 2026-09-08 carry 0.4165 in meta "
                    "rov_drawn_geometry.jaw_centre_m / policy.tcp_body_flu_m.]",
    "JAW_HALF_M": "[스펙: tools/gen_gripper_variant.py:48 JAW_SIZE "
                  "half-extents — 60 mm long, 6 mm thick, 24 mm tall]",
    "JAW_CLOSED_HALF_GAP_M": "[스펙: tools/gen_gripper_variant.py:49 JAW_Y]",
    "JAW_TRAVEL_M": "[스펙: tools/gen_gripper_variant.py:47 JAW_TRAVEL, "
                    "which cites the vendor's 62 mm opening. The SIM's own "
                    "jaw boxes do not reproduce that: half-gap 0.008 minus "
                    "half-thickness 0.003 is a 10 mm clear gap closed and "
                    "72 mm open, not 0 -> 62. This module draws the sim's "
                    "geometry, so the drawn jaws open 10 -> 72 mm.]",
    "THRUSTERS_M": "[유도: bluerov2_mujoco_marinegym/bluerov_heavy_gripper.xml "
                   "thruster_0..7 site pos/quat, shifted by the composite-COM "
                   "offset (0.03489, 0.00099, -0.02579) into the vehicle-COM "
                   "frame; axis = quat applied to local +X, the thrust "
                   "direction the model and thrusters.py:206 both use]",
    "THRUSTER_R_M": "[유도: the operator-tape footprint half-width 0.5334/2 "
                    "minus the vertical thrusters' y 0.220 = 0.0467, i.e. "
                    "the assumption that those shrouds ARE the width. No "
                    "T200 shroud dimension exists in this repo, so there is "
                    "no independent check — and a CAD footprint of 0.5749 "
                    "circulates too, which would instead give 0.0675. The "
                    "0.0467 is kept because it is the tape the panel already "
                    "draws its rectangle from.]",
    "THRUSTER_L_M": "[예측] drawing only — no source.",
    "CAM_T_FLU_M": "[유도: config/hw_nav.yaml `cam_t_flu`, itself derived "
                   "from the operator's CAD of the re-mounted C3 "
                   "([측정] 2026-09-02: lens 275.014 mm UP and 110.664 mm "
                   "BACK of the jaw-pair centre). This is the value that "
                   "REPLACED a sign-flipped one — the old [0.23949, 0.00547, "
                   "-0.15537] put the lens 0.155 m BELOW the COM and threw "
                   "the vehicle 0.46 m into the air in this very view. "
                   "2026-09-08: the CAD's 110.664 mm lens->jaw is CONTRADICTED "
                   "by the C3 image (measured 196 mm, LENS_TO_GRIP_FLU_M), so "
                   "either that offset or this lens x/z is wrong. The lens "
                   "value is KEPT — it is the record boundary for every "
                   "recorded position — but its x/z are now SUSPECT (the lens "
                   "may be ~9 cm further aft and ~5 cm higher; KNOWN_ISSUES "
                   "2026-09-08 'cam_t_flu x/z'). The jaw is anchored to the "
                   "lens, so the drawn/controller TCP is right either way; "
                   "the HULL is what moves if this value changes.]",
    "CAM_DIMS_M": "[스펙: bluerov2_mujoco_marinegym/"
                  "compute_payload_inertia.py:64 C3_DIMS — the HOUSING, so "
                  "still valid across the re-mount that moved the position]",
    "CAM_TILT_DEG": "[측정 2026-09-02: config/hw_nav.yaml `cam_tilt_deg`, "
                    "the re-mount angle the localizer actually applies. The "
                    "panel takes this from the live nav config when it has "
                    "one, so the picture cannot disagree with the solver.]",
    "CAM_RAY_M": "[예측] a pointer showing WHICH WAY the lens looks. Not a "
                 "range, not a frustum, and deliberately not scaled to the "
                 "C3's FOV — that would be a claim about what is visible.",
    "HEADING_FWD_M": "[예측] drawing only — how far the heading line runs "
                     "AHEAD of the COM (operator request 2026-09-07). Not a "
                     "reach, a sight line.",
    "HEADING_AFT_M": "[예측] drawing only — how far the heading line runs "
                     "BEHIND the COM (operator request 2026-09-07).",
    "JAW_DRAW_OPEN_FRAC": "[예측] drawing only. The station has NO jaw "
                          "feedback (PayloadState.gripper_fb is None on the "
                          "hardware ROV), so the drawn opening is a "
                          "schematic constant and NOT the jaw's position.",
}


def rov_geometry_meta() -> dict:
    """The drawn geometry + its provenance, for the run meta. A recorded run
    should be able to say which picture of the vehicle it was flown with —
    the numbers here are the ones a screenshot is measured against."""
    return {
        "frame": "body FLU (x forward, y left, z up), origin = vehicle COM",
        "hull_l_m": HULL_L_M, "hull_w_m": HULL_W_M,
        "hull_h_m": HULL_H_M, "hull_z_m": HULL_Z_M,
        "underside_z_m": UNDERSIDE_Z_M,
        "grip_centre_m": list(GRIP_CENTRE_M),
        "grip_r_m": GRIP_R_M, "grip_l_m": GRIP_L_M,
        "jaw_centre_m": list(JAW_CENTRE_M),
        "lens_to_grip_flu_m": list(LENS_TO_GRIP_FLU_M),
        "jaw_half_m": list(JAW_HALF_M),
        "jaw_closed_half_gap_m": JAW_CLOSED_HALF_GAP_M,
        "jaw_travel_m": JAW_TRAVEL_M,
        "cam_t_flu_m": list(CAM_T_FLU_M), "cam_dims_m": list(CAM_DIMS_M),
        "cam_tilt_deg": CAM_TILT_DEG, "cam_ray_m": CAM_RAY_M,
        "heading_fwd_m": HEADING_FWD_M, "heading_aft_m": HEADING_AFT_M,
        "thrusters_m": [[list(c), list(a)] for c, a in THRUSTERS_M],
        "thruster_r_m": THRUSTER_R_M, "thruster_l_m": THRUSTER_L_M,
        "jaw_draw_open_frac": JAW_DRAW_OPEN_FRAC,
        "jaw_opening_is_live": False,
        "provenance": dict(GEOMETRY_PROVENANCE),
    }


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def body_to_map(p_flu, origin_ned, yaw: float):
    """Body FLU -> MAP NED. THE ONLY place the two sign flips happen.

    ``origin_ned`` is where the body origin (the vehicle COM) sits in the map
    — i.e. exactly the position the panel already draws its dot at — and
    ``yaw`` is the NED heading, so at yaw 0 body +x points along map +x.

    Body y is LEFT and map y is EAST; body z is UP and map z is DOWN. Both
    flips are here and nowhere else.
    """
    x, y, z = float(p_flu[0]), float(p_flu[1]), float(p_flu[2])
    ca, sa = math.cos(yaw), math.sin(yaw)
    # y_flu is LEFT -> the starboard-positive coordinate the yaw rotation
    # expects is its negation.
    dr = -y
    return (float(origin_ned[0]) + ca * x - sa * dr,
            float(origin_ned[1]) + sa * x + ca * dr,
            float(origin_ned[2]) - z)          # map z is DOWN


def _prism(centre, axis, half_len, radius, half2=None):
    """An oriented box: half-extents ``(half_len, radius, half2 or radius)``
    along ``(axis, perp1, perp2)`` about ``centre``.

    With ``half2`` omitted the section is SQUARE, which is what the thruster
    shrouds use — for the reason `parts()` gives for the gripper tube: at
    every zoom this panel reaches a 47 mm radius is a handful of pixels, and
    eight more vertices per thruster per frame would buy nothing a human eye
    could see. The camera passes ``half2`` because its housing really is
    165 mm wide and 89 mm tall, and drawing that square made it a third
    taller than the C3 is.
    """
    half2 = radius if half2 is None else float(half2)
    ax = list(float(v) for v in axis)
    n = math.sqrt(sum(v * v for v in ax)) or 1.0
    ax = [v / n for v in ax]
    # A perpendicular that never degenerates: cross with whichever world axis
    # the thrust axis is least aligned with.
    ref = (0.0, 0.0, 1.0) if abs(ax[2]) < 0.9 else (1.0, 0.0, 0.0)
    p1 = [ax[1] * ref[2] - ax[2] * ref[1],
          ax[2] * ref[0] - ax[0] * ref[2],
          ax[0] * ref[1] - ax[1] * ref[0]]
    n1 = math.sqrt(sum(v * v for v in p1)) or 1.0
    p1 = [v / n1 for v in p1]
    p2 = [ax[1] * p1[2] - ax[2] * p1[1],
          ax[2] * p1[0] - ax[0] * p1[2],
          ax[0] * p1[1] - ax[1] * p1[0]]
    v = []
    for sa in (-1, 1):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                v.append(tuple(
                    centre[k] + sa * half_len * ax[k]
                    + s1 * radius * p1[k] + s2 * half2 * p2[k]
                    for k in range(3)))
    # Same index scheme as _box: 4*(sa>0) + 2*(s1>0) + (s2>0), so the face
    # table below is the same table with (ax, p1, p2) in place of (x, y, z).
    faces = [
        ((0, 1, 3, 2), tuple(-a for a in ax)),
        ((4, 6, 7, 5), tuple(ax)),
        ((0, 4, 5, 1), tuple(-a for a in p1)),
        ((2, 3, 7, 6), tuple(p1)),
        ((0, 2, 6, 4), tuple(-a for a in p2)),
        ((1, 5, 7, 3), tuple(p2)),
    ]
    return v, faces


def _box(centre, half):
    """The 8 corners of an axis-aligned body-frame box, and its 6 faces as
    index quads wound so a face's vertices are in order around it."""
    cx, cy, cz = centre
    hx, hy, hz = half
    v = [(cx + sx * hx, cy + sy * hy, cz + sz * hz)
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    # index = 4*(sx>0) + 2*(sy>0) + (sz>0)
    # (indices, OUTWARD normal in body FLU). The normal is carried rather
    # than derived from the winding because these are axis-aligned boxes and
    # a literal is both cheaper and impossible to get backwards — and a
    # flipped normal is invisible until the shading is inside out.
    faces = [
        ((0, 1, 3, 2), (-1.0, 0.0, 0.0)),      # aft
        ((4, 6, 7, 5), (1.0, 0.0, 0.0)),       # forward
        ((0, 4, 5, 1), (0.0, -1.0, 0.0)),      # starboard (body y is LEFT)
        ((2, 3, 7, 6), (0.0, 1.0, 0.0)),       # port
        ((0, 2, 6, 4), (0.0, 0.0, -1.0)),      # bottom
        ((1, 5, 7, 3), (0.0, 0.0, 1.0)),       # top
    ]
    return v, faces


def silhouette_xy(verts):
    """The top-down outline of one part: the 2-D convex hull of its vertices
    in body x-y, counter-clockwise.

    Needed because four of the eight thrusters are ROTATED 45 deg about z, so
    their footprint is a diamond and their axis-aligned bounding box is not
    it — drawing the box made the vectored thrusters read as bigger square
    ones and hid the whole point of showing them, which is which way they
    push. Monotone chain; eight points, so the O(n log n) is free.
    """
    pts = sorted({(round(float(v[0]), 9), round(float(v[1]), 9))
                  for v in verts})
    if len(pts) < 3:
        return pts

    def _half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) > 0:
                    break
                out.pop()
            out.append(q)
        return out[:-1]

    return _half(pts) + _half(list(reversed(pts)))


def camera_axis_flu(tilt_deg: float = CAM_TILT_DEG):
    """The optical axis in body FLU: +x pitched DOWN by ``tilt_deg``."""
    t = math.radians(float(tilt_deg))
    return (math.cos(t), 0.0, -math.sin(t))


def heading_line_body(fwd_m: float = HEADING_FWD_M,
                      aft_m: float = HEADING_AFT_M):
    """The two ends of the heading line, in body FLU: on the vehicle's
    x-axis through the COM, ``aft_m`` behind it and ``fwd_m`` ahead of it.
    Both at z = 0 — the line lies in the vehicle's own horizontal plane, so
    in 3-D it is distinct from the optical-axis ray (which pitches down) and
    from above the two are collinear (which is why the top-down marker
    draws only this one; `TrajectoryView._draw_rov_heading`)."""
    return ((-float(aft_m), 0.0, 0.0), (float(fwd_m), 0.0, 0.0))


def parts(jaw_open_frac: float = JAW_DRAW_OPEN_FRAC,
          hull_l: float = HULL_L_M, hull_w: float = HULL_W_M,
          hull_h: float = HULL_H_M, cam_tilt_deg: float = CAM_TILT_DEG):
    """The vehicle as a list of parts, in body FLU metres.

    Each part is ``{"name", "verts", "faces", "kind"}`` where ``verts`` are
    body-frame points and ``faces`` is a list of ``(indices, normal)`` with
    the OUTWARD normal in body FLU. ``kind`` is what the drawing
    should do with it: ``"hull"`` is the big box (and the ONLY part whose
    top-down outline is the existing footprint rectangle, so the two modes
    agree), ``"tube"`` and ``"jaw"`` are the gripper.

    ``hull_l``/``hull_w`` are parameters because the window sets the panel's
    footprint from hw_nav.yaml and the picture must follow that, not a
    second copy of it.
    """
    out = []

    hv, hf = _box((0.0, 0.0, UNDERSIDE_Z_M + hull_h / 2.0),
                  (hull_l / 2.0, hull_w / 2.0, hull_h / 2.0))
    out.append({"name": "hull", "kind": "hull", "verts": hv, "faces": hf})

    # The gripper tube as a square prism rather than a cylinder: at every
    # zoom this panel is ever at, an 18 mm radius is at most a couple of
    # pixels, so a circular cross-section would cost eight more vertices per
    # frame to draw the same two pixels.
    gv, gf = _box(GRIP_CENTRE_M, (GRIP_L_M / 2.0, GRIP_R_M, GRIP_R_M))
    out.append({"name": "tube", "kind": "tube", "verts": gv, "faces": gf})

    for i, (c, ax) in enumerate(THRUSTERS_M):
        tv, tf = _prism(c, ax, THRUSTER_L_M / 2.0, THRUSTER_R_M)
        out.append({"name": f"thruster_{i}", "kind": "thruster",
                    "verts": tv, "faces": tf})

    # The camera housing, pitched with the lens. Its long axis is the
    # OPTICAL axis, so `_prism` orients it the same way the ray is drawn and
    # the two can never disagree about which way the C3 faces.
    # (optical depth, width, height) onto (axis, perp1, perp2). `_prism`
    # picks perp1 = the body y axis for any axis tilted about y, which is
    # what the C3's tilt is, so width lands on width.
    cv, cf = _prism(CAM_T_FLU_M, camera_axis_flu(cam_tilt_deg),
                    CAM_DIMS_M[0] / 2.0, CAM_DIMS_M[1] / 2.0,
                    CAM_DIMS_M[2] / 2.0)
    out.append({"name": "camera", "kind": "camera", "verts": cv, "faces": cf})

    frac = max(0.0, min(1.0, float(jaw_open_frac)))
    gap = JAW_CLOSED_HALF_GAP_M + frac * JAW_TRAVEL_M
    for side, sign in (("jaw_left", +1.0), ("jaw_right", -1.0)):
        c = (JAW_CENTRE_M[0], JAW_CENTRE_M[1] + sign * gap, JAW_CENTRE_M[2])
        jv, jf = _box(c, JAW_HALF_M)
        out.append({"name": side, "kind": "jaw", "verts": jv, "faces": jf})
    return out


def tcp_body_m():
    """The point the diffusion policy's plans are composed at: the jaw-pair
    centre. Identical to hw_mpc.yaml `policy.tcp_body_flu_m` — the test
    asserts that, because two copies of a TCP is exactly how a picture ends
    up disagreeing with the controller it illustrates."""
    return JAW_CENTRE_M


def tcp_body_for_meta(meta) -> tuple[tuple[float, float, float], str]:
    """The TCP a RECORDED run composed its plans at, body FLU, from its meta.

    ``plans.jsonl`` ``p_tcp`` / ``anchor_tcp`` and the meta's ``T_body_tcp``
    were recorded with the TCP the station carried THAT day, and that point
    moved on 2026-09-08 (jaw anchored to the lens: 0.4165 -> 0.502 m ahead
    of the COM). An offline tool that draws the TCP dot from THIS module on
    an older run puts it 8.5 cm ahead of the run's own plan polyline — two
    record generations in one picture. So the dot follows the run's meta
    (``rov_drawn_geometry.jaw_centre_m``, else ``policy.tcp.tcp_body_flu_m``)
    and falls back to the live ``tcp_body_m()`` only for a run that recorded
    neither. Returns ``(xyz, source)``; ``source`` says which it was, so the
    tool can print it. Only the DOT is per-run: the jaw POLYGONS (``parts``)
    are still this module's, i.e. today's vehicle.
    """
    meta = meta if isinstance(meta, dict) else {}
    geo = meta.get("rov_drawn_geometry")
    v = geo.get("jaw_centre_m") if isinstance(geo, dict) else None
    src = "meta rov_drawn_geometry.jaw_centre_m"
    if v is None:
        pol = meta.get("policy")
        tcp = pol.get("tcp") if isinstance(pol, dict) else None
        v = tcp.get("tcp_body_flu_m") if isinstance(tcp, dict) else None
        src = "meta policy.tcp.tcp_body_flu_m"
    try:
        xyz = tuple(float(x) for x in v)
    except (TypeError, ValueError):
        xyz = ()
    if len(xyz) != 3:
        return tuple(float(x) for x in tcp_body_m()), \
            "live rov_shape (the run recorded no TCP)"
    return xyz, src


def footprint(p_ned, yaw: float, hull_l: float = HULL_L_M,
              hull_w: float = HULL_W_M):
    """The hull's top-down outline in MAP coordinates — the same four
    corners `TrajectoryView._hull` has always returned, expressed through
    `body_to_map` so the flat marker and the 3-D body can never drift apart.
    """
    hl, hw = hull_l / 2.0, hull_w / 2.0
    # Wound to reproduce `TrajectoryView._hull`'s historical order EXACTLY
    # (port nose, starboard nose, starboard tail, port tail): the nose mark
    # is drawn from corner 0 to corner 1, and the test asserts corner-for-
    # corner equality — the marker must not move by a millimetre when the
    # picture gains a gripper. Note the flip: `_hull`'s `dy` is the
    # starboard-positive map coordinate, so `dy = -hw` is body y = +hw.
    corners_flu = ((hl, hw, 0.0), (hl, -hw, 0.0),
                   (-hl, -hw, 0.0), (-hl, hw, 0.0))
    return [body_to_map(c, (p_ned[0], p_ned[1], 0.0), yaw)[:2]
            for c in corners_flu]
