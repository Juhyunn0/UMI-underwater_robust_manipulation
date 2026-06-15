"""DOB-MPC for the marinegym BlueROV2 sim.

The validated EAOB + NMPC math (fossen.py, eaob.py, mpc.py) is reused verbatim
from the standalone `bluerov2_mujoco_dobmpc` package (paper: Hu et al., JMSE
2024). Only two things are marinegym-specific:

  * params.py  -- rebuilt from marinegym's BlueROV.yaml / bluerov.xml so the
    prediction model matches THIS plant (damping sign-flipped to marinegym's
    convention; M_A, inertia, buoyancy, ZG from the YAML). Only the true
    current/wave/kick disturbance is then left as w.
  * frames.py  -- FLU<->NED adapter. The observer/MPC run in the paper's
    NED/FRD; marinegym is FLU/z-up. A fixed S=diag(1,-1,-1) conjugation maps
    state in and the 4-DOF wrench out.

The controller wrapper lives one level up in dobmpc_controller.py and matches
the PoseController interface so teleop.py drives it unchanged.
"""
