"""rov_gui.control — closed-loop MPC on the real vehicle, from AprilTags in.

The pipeline this package adds to the station (all inside the ONE process that
owns the C3 and the command sink, because both are single-owner):

    C3 colour frames ──► tagnav (detect + PnP against a locked tag map)
                          │  NavFix: vehicle pose in the NED world frame
    ArduSub MAVLink  ──►  │  VehicleImu: attitude / rates / accel / depth
                          ▼
                 state_assembler (eta, nu, nudot in NED/FRD, 20 Hz)
                          ▼
                 mpc_bridge (the sim's EAOB + acados NMPC, reused verbatim)
                          ▼
                 allocation (NED wrench -> PilotInput axes)
                          ▼
                 the EXISTING MavlinkCommandSink (MANUAL_CONTROL, all gates)

Layering rule: everything except ``workers`` is Qt-free plain Python, and
NOTHING in this package imports ``dobmpc`` / ``casadi`` / ``acados_template``
at module scope — only ``mpc_bridge`` does, inside functions — so the demo
backend and the offline tests run on a machine with no acados at all.

The controller mathematics is NOT reimplemented here. ``dobmpc/`` (verified
MuJoCo-free) is imported as-is from ``bluerov2_mujoco_marinegym/``; the SQUARE
reference generators are byte-for-byte copies (see reference.py provenance) —
the rectangle, the line and the circle beside them are hardware additions with
no simulator original, so a result flown on one of those inherits none of the
sim's validation;
the frame conversions go through ``dobmpc.frames`` so this package inherits
the sim-validated FLU<->NED path instead of growing a second one.
"""
