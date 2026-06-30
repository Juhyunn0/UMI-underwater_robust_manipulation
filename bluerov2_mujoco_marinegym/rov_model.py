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
}

MODEL = os.environ.get("ROV_MODEL", "heavy").strip().lower()
if MODEL not in _MODELS:
    raise ValueError(f"ROV_MODEL={MODEL!r} not in {list(_MODELS)}")

_CFG = _MODELS[MODEL]
_HERE = os.path.dirname(os.path.abspath(__file__))

XML_NAME = _CFG["xml"]
YAML_NAME = _CFG["yaml"]
XML_PATH = os.path.join(_HERE, XML_NAME)
YAML_PATH = os.path.join(_HERE, "marinegym_assets", YAML_NAME)
MASS = _CFG["mass"]
INERTIA = tuple(_CFG["inertia"])           # (Ix, Iy, Iz)
VOLUME = _CFG["volume"]
N_THRUSTERS = _CFG["n_thrusters"]
FULLY_ACTUATED = _CFG["fully_actuated"]
