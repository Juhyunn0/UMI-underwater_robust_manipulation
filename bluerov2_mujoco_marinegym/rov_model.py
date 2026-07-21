"""Single source of truth for WHICH BlueROV variant the whole stack uses.

Select with the env var `ROV_MODEL` (default "heavy"):

    python teleop.py --square --ctrl dobmpc --disturb   # heavy (default, rank-6)
    ROV_MODEL=bluerov2 python -m dobmpc.eval_dp         # vectored-6 (rank-5)

Two variants, both from the MarineGym assets (values verified directly from the
USD: external/MarineGym/.../usd/{BlueROV,BlueROVHeavy}/):

  * **bluerov2** — the standard vectored-6. 6 thrusters (4 horizontal at z=-0.0725
    + 2 vertical). Allocation B is 6x6, **rank 5**: pitch is NOT independently
    controllable (under-actuated). The NMPC commands u=[X,Y,Z,N] (NU=4) and leaves
    pitch to its trim (option-(b) surge-pitch handling).

  * **heavy** — BlueROV2 Heavy. 8 thrusters (the same 4 horizontal + 4 vertical at
    the corners). Allocation B is 6x8, **rank 6 = fully actuated**: roll AND pitch
    ARE directly controllable. The NMPC commands the full wrench u=[X,Y,Z,K,M,N]
    (NU=6); no option-(b) surge cap is needed.

Heavy keeps the SAME hydro coefficients (added mass, linear/quadratic damping) and
the SAME T200 thrusters as the standard BlueROV2 -- only the mass, inertia, buoyant
volume, and the thruster layout differ. (The MarineGym Heavy yaml lists a weaker
force_constants 0.8e-7; we keep the validated T200 curve since the physical thruster
is unchanged -- see docs/03_THRUSTERS.md.)

Everything else (params.py, the MJCF loaders, hydro.py, thrusters.py allocation)
reads from here so the plant and the controller can never disagree on the model.
"""
import os

_MODELS = {
    "bluerov2": dict(
        xml="bluerov.xml",
        yaml="BlueROV.yaml",
        mass=11.2,
        inertia=(0.30375, 0.626, 0.5769),
        volume=0.0113459,
        n_thrusters=6,
        fully_actuated=False,
    ),
    "heavy": dict(
        xml="bluerov_heavy.xml",
        yaml="BlueROVHeavy.yaml",
        mass=11.5,
        # The MarineGym/farol Heavy USD ships inertia (0.21, 0.245, 0.245), but that is a
        # HAND-TUNED Gazebo-stability literal (farol bluerov_heavy_vehicle/urdf/base.xacro),
        # NOT physical. Instead this is DERIVED from the BlueROV2 inertia by adding the
        # parallel-axis contribution of the vertical-thruster layout change (BlueROV2's 2
        # near-centre verticals -> Heavy's 4 corner verticals; the 4 horizontals are
        # identical and cancel), thrusters as point masses of 0.15 kg (the +0.3 kg / +2
        # thrusters mass budget). Reproduce/adjust with compute_heavy_inertia.py.
        # I_heavy = I_bluerov2 + [+0.02538, +0.00865, +0.03402]. See CONTROL_METHODOLOGY.md.
        inertia=(0.3291, 0.6347, 0.6109),
        volume=0.0116499,
        n_thrusters=8,
        fully_actuated=True,
    ),
    "heavy_gripper": dict(
        # heavy + Newton Subsea Gripper (jaws articulated, ctrl index 8) + MarineSitu C3
        # stereo camera, rigidly attached. Rigid-body numbers COMPOSED (parallel-axis)
        # from the verified vendor masses -- gripper 0.524 kg (2x0.030 kg jaws as child
        # bodies), C3 1.700 kg -- by compute_payload_inertia.py; the MJCF is GENERATED
        # from bluerov_heavy.xml by gen_gripper_variant.py (rerun after edits there).
        # The generated frame is RE-ORIGINED at the composite COM (origin==COM, like
        # heavy) -- required by the dobmpc predictor / params.ZG_MASS=0 / hydro, which
        # all assume it; without it the NMPC closed loop destabilizes. mass = total
        # subtree (incl. jaws); inertia = TOTAL composite about the total COM
        # (Ixz=+0.064, 16.8% of Ixx, DROPPED by the diagonal-inertia constraint --
        # see gen_gripper_variant.py / KNOWN_ISSUES). volume adds the payloads'
        # displaced water -> net buoyancy ~ -5.7 N (SINKS: real payload, no trim
        # foam, per the 2026-07-12 decision). Hydro added-mass/damping stay the
        # heavy set (increments documented in the YAML). C3 placement measured from
        # the user's Onshape assembly 2026-07-19 (front-bottom, lens forward level).
        xml="bluerov_heavy_gripper.xml",
        yaml="BlueROVHeavyGripper.yaml",
        mass=13.7240,
        inertia=(0.38154, 0.77780, 0.70954),
        volume=0.0131815,
        n_thrusters=8,
        fully_actuated=True,
    ),
    "heavy_c3": dict(
        # heavy + MarineSitu C3 stereo camera on its C3-BR bracket = EXACTLY the lab's
        # Onshape assembly (the Newton gripper is NOT in Onshape yet, so it is absent
        # here; heavy_gripper is the future config for when it exists in CAD). Rigid-body
        # numbers COMPOSED (parallel-axis) from the vehicle + C3 vendor mass 1.700 kg by
        # compute_payload_inertia.compose_c3(); the MJCF is GENERATED from bluerov_heavy.xml
        # by gen_c3_variant.py (rerun after edits there). Frame RE-ORIGINED at the composite
        # COM (origin==COM, like heavy). C3 placement measured from Onshape 2026-07-19
        # (front-bottom, lens forward level). Bracket is visual-only (mass unknown). Dropped
        # Ixz = +0.046 (12.4% of Ixx, KNOWN_ISSUES). volume adds the C3's displaced water ->
        # net buoyancy ~ -3.1 N (SINKS: payload, no trim foam).
        xml="bluerov_heavy_c3.xml",
        yaml="BlueROVHeavyC3.yaml",
        mass=13.2000,
        inertia=(0.37014, 0.73153, 0.67460),
        volume=0.0129237,
        n_thrusters=8,
        fully_actuated=True,
    ),
}

MODEL = os.environ.get("ROV_MODEL", "heavy").strip().lower()
if MODEL not in _MODELS:
    raise ValueError(f"ROV_MODEL={MODEL!r} not in {list(_MODELS)}")

_CFG = _MODELS[MODEL]
_HERE = os.path.dirname(os.path.abspath(__file__))

# Opt-in real-pool AprilTag floor (VISUAL ONLY; built by gen_pool_apriltags.py).
# POOL_TAGS=1 swaps XML_PATH to a wrapper scene = the SAME ROV + the tag_floor.xml
# fragment (seabed + tag36h11 grid + a translucent water surface). The added geoms
# are contype=0 conaffinity=0 and MuJoCo's fluid model is OFF, so dynamics are
# byte-for-byte unchanged. Unset -> the plain model loads exactly as before, and
# every loader that reads RM.XML_PATH (teleop.py, eval_dp, run_compare, test_*/verify_*)
# picks this up for free.
_POOL_TAGS = os.environ.get("POOL_TAGS", "0").strip().lower() in ("1", "true", "yes", "on")
_POOL_WRAP = {"bluerov2": "scene_bluerov_tags.xml", "heavy": "scene_bluerov_heavy_tags.xml",
              "heavy_gripper": "scene_bluerov_heavy_gripper_tags.xml",
              "heavy_c3": "scene_bluerov_heavy_c3_tags.xml"}

XML_NAME = _POOL_WRAP[MODEL] if _POOL_TAGS else _CFG["xml"]
YAML_NAME = _CFG["yaml"]
XML_PATH = os.path.join(_HERE, XML_NAME)
YAML_PATH = os.path.join(_HERE, "marinegym_assets", YAML_NAME)
MASS = _CFG["mass"]
INERTIA = tuple(_CFG["inertia"])           # (Ix, Iy, Iz)
VOLUME = _CFG["volume"]
N_THRUSTERS = _CFG["n_thrusters"]
FULLY_ACTUATED = _CFG["fully_actuated"]
