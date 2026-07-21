#!/usr/bin/env python3
"""Baseline PD/PID setpoint (go-to-origin) controller for the BlueROV2 (FLU).

Drives the vehicle to a world setpoint (default: global origin) + yaw reference in
the MarineGym FLU model. Fully FLU — no NED (that conversion is only the Phase-7
DOB-MPC's boundary). Commands surge/sway/heave/yaw; NEVER pitch (rank-5
underactuated), and leaves roll/pitch to passive buoyancy restoring.

Design (reviewed by control-theory-advisor):
  * World-frame position PD; rotate ONLY the final force to body (avoids the
    rotating-stiffness / crabbing issue with anisotropic gains).
  * D term uses world linear velocity (data.qvel[:3]); yaw D uses body yaw rate.
  * Net-buoyancy feedforward (the known +1.1 N), and a GATED anti-windup integral
    (integrate only near the setpoint) for the unknown current bias.
  * Surge force saturated in WRENCH space + slew-rate limited (per-thruster
    clipping would distort the wrench direction); optional soft pitch guard.
Reuses thrusters.py allocation (allocation_matrix / set_wrench_command).
See docs/03_THRUSTERS.md (rank-5, My≈-0.0725·Fx surge->pitch coupling).
"""
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thrusters as T
import rov_model as RM


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# Gains are per-variant (rov_model). Translational gains are WORLD-frame.
#
# heavy — analytic pole placement (designed 2026-07-02, applied 2026-07-03; see
# docs/CONTROL_METHODOLOGY.md). Per-axis hover linearization m_eff·a = F − d_lin·v;
# closed-loop char poly m·s³ + (d+Kd)·s² + Kp·s + Ki matched to the target
# (s² + 2ζωn·s + ωn²)(s + αωn):
#   Kp = m_eff·(1+2ζα)·ωn²,  Kd = m_eff·(2ζ+α)·ωn − d_lin,  Ki = m_eff·α·ωn³
# Design point: ωn=2.0 rad/s translation (above the JONSWAP energy band
# 0.45–1.2 rad/s), ωn=3.0 yaw, ζ=0.9, α=0.2. Horizontal gains are ISOTROPIC
# (designed on the sway m_eff=24.2 kg; body-x then realizes ωn≈2.78, ζ≈1.0):
# world-frame PD only commutes with yaw when kp_x=kp_y, and the primary compare
# scenario is square + heading_follow. Companion values (NOT optional): surge
# f_max 30 N (the old 6 N cap was a rank-5 pitch-coupling relic and would bind
# ωn to ≤0.54), e_gate 0.15 m, surge_slew 120 N/s, and the yaw rate reference
# feed-forward r_ref in compute(). Sim-only validity (ideal thrusters, perfect
# state, 500 Hz) — derate ωn to 1.0–1.5 and filter D for hardware.
GAINS_HEAVY = dict(
    kp=(131.6, 131.6, 141.8),   # surge, sway, heave  [N/m]
    kd=(90.6, 90.6, 99.1),      #                     [N·s/m]
    ki=(38.7, 38.7, 41.7),      #                     [N/(m·s)]
    i_max=(4.0, 5.0, 5.0),      # |Ki*integral| clamp [N]
    f_max=(30.0, 30.0, 30.0),   # wrench-space force saturation [N] (rank-6: no
                                # surge-pitch coupling, so no surge derate)
    surge_slew=120.0,           # surge force slew-rate limit [N/s]
    yaw_kp=8.95, yaw_kd=4.32, yaw_ki=3.95, yaw_i_max=2.0, mz_max=10.0,  # yaw [N·m]
    e_gate=0.15, yaw_gate=0.2,  # integrate only when |error| < gate (m, rad)
    pitch_guard_deg=15.0,       # soft-scale surge above this |pitch| (deg)
)

# bluerov2 (vectored-6, rank-5) — legacy hand-tuned set (2026-06-14), kept as-is:
# the pole-placement design above was derived and verified for heavy only, and
# the 6 N surge cap here is load-bearing on rank-5 (a steady Fx balances the weak
# pitch restoring 1.11 N·m at sin(theta)=0.0725·Fx/1.11, so 6 N -> ~23°).
GAINS_BLUEROV2 = dict(
    kp=(6.0, 12.0, 20.0),       # surge, sway, heave  [N/m]
    kd=(8.0, 12.0, 22.0),       #                     [N·s/m]  (heave: true still-
                                # water zeta≈0.60 with the full m_eff=26.07 kg)
    ki=(1.0, 1.5, 1.2),         #                     [N/(m·s)]
    i_max=(4.0, 5.0, 5.0),      # |Ki*integral| clamp [N]
    f_max=(6.0, 30.0, 30.0),    # surge kept low — see rank-5 note above
    surge_slew=30.0,            # surge force slew-rate limit [N/s] (softens pitch kick)
    yaw_kp=5.0, yaw_kd=3.0, yaw_ki=0.5, yaw_i_max=2.0, mz_max=10.0,  # yaw [N·m]
    e_gate=0.5, yaw_gate=0.2,   # integrate only when |error| < gate (m, rad)
    pitch_guard_deg=15.0,       # soft-scale surge above this |pitch| (deg)
)

# heavy_gripper — the SAME pole-placement design re-evaluated at the payload masses
# (heavy + Newton gripper + MarineSitu C3: m=13.724 kg, Iz=0.6906; added-mass set
# unchanged). Same design point (ωn=2.0 translation / 3.0 yaw, ζ=0.9, α=0.2):
#   sway  m_eff = 13.724+12.7  = 26.42 kg -> kp=143.7 kd=99.5  ki=42.3 (isotropic hor.)
#   heave m_eff = 13.724+14.57 = 28.29 kg -> kp=153.9 kd=108.0 ki=45.3
#   yaw   I_eff = 0.6906+0.12  = 0.811    -> kp=9.92  kd=4.79  ki=4.38
# The static -5.7 N net buoyancy (payload sinks) is carried by the buoyancy_ff term
# in compute(), not by these gains.
GAINS_HEAVY_GRIPPER = dict(
    kp=(143.7, 143.7, 153.9),   # surge, sway, heave  [N/m]
    kd=(99.5, 99.5, 108.0),     #                     [N·s/m]
    ki=(42.3, 42.3, 45.3),      #                     [N/(m·s)]
    i_max=(4.0, 5.0, 5.0),      # |Ki*integral| clamp [N]
    f_max=(30.0, 30.0, 30.0),   # rank-6: no surge derate
    surge_slew=120.0,           # surge force slew-rate limit [N/s]
    yaw_kp=9.92, yaw_kd=4.79, yaw_ki=4.38, yaw_i_max=2.0, mz_max=10.0,  # yaw [N·m]
    e_gate=0.15, yaw_gate=0.2,  # integrate only when |error| < gate (m, rad)
    pitch_guard_deg=15.0,       # soft-scale surge above this |pitch| (deg)
    # ACTIVE roll/pitch leveling (rank-6): the payload's static attitude torque
    # (forward COM + jaw weight, ~0.2-0.5 N·m) rivals the passive restoring
    # B·coBM ≈ 1.1 N·m/rad, so without this the attitude walks off and flips.
    # PD pole placement at ωn=3, ζ=0.9 on I_eff = I + K'/M': roll 0.481, pitch 0.859.
    rp_kp=(4.3, 7.7), rp_kd=(2.6, 4.6), rp_max=8.0,   # roll, pitch [N·m]
)

# Per-variant gains; unknown variants fall back on the structural property that
# motivated the original split (the surge cap is a rank-5/under-actuation artifact),
# so a future variant fails safe, not silently.
_GAINS_BY_MODEL = {"bluerov2": GAINS_BLUEROV2, "heavy": GAINS_HEAVY,
                   "heavy_gripper": GAINS_HEAVY_GRIPPER}
DEFAULT_GAINS = _GAINS_BY_MODEL.get(
    RM.MODEL, GAINS_HEAVY if RM.FULLY_ACTUATED else GAINS_BLUEROV2)


class PoseController:
    def __init__(self, model, mode="pid", setpoint=(0.0, 0.0, 0.0), yaw_ref=0.0,
                 buoyancy_ff=None, body="base_link", gains=None, dt=None, actuator=None):
        self.model = model
        self.mode = mode
        self.actuator = actuator                         # optional realistic thrusters
        self.use_i = (mode == "pid")
        self.p_ref = np.asarray(setpoint, float)
        self.yaw_ref = float(yaw_ref)
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.B, _ = T.allocation_matrix(model)          # constant body geometry
        self.dt = float(model.opt.timestep) if dt is None else float(dt)
        g = dict(DEFAULT_GAINS)
        if gains:
            g.update(gains)
        self.g = g
        # net buoyancy feedforward (world +z): a Hydrodynamics instance, a float
        # (net N), or None (0). Net = buoyancy - weight (~+1.1 N here).
        if buoyancy_ff is None:
            self.net_buoy = 0.0
        elif isinstance(buoyancy_ff, (int, float)):
            self.net_buoy = float(buoyancy_ff)
        else:
            self.net_buoy = float(getattr(buoyancy_ff, "buoyancy", 0.0)
                                  - getattr(buoyancy_ff, "weight", 0.0))
        self.reset()

    def reset(self):
        self._I = np.zeros(3)
        self._I_yaw = 0.0
        self._fx_prev = 0.0
        self.v_ref = np.zeros(3)             # world reference velocity (trajectory FF)
        self.r_ref = 0.0                     # reference yaw rate (heading-follow FF)
        self.commanded = np.zeros(6)
        self.realized = np.zeros(6)
        if getattr(self, "actuator", None) is not None:
            self.actuator.reset()

    def set_target(self, p_ref=None, yaw_ref=None, v_ref=None, r_ref=None,
                   yaw_target=None):
        """Update the (possibly moving) setpoint. v_ref is the reference world velocity
        used as a feed-forward in the derivative term, for trajectory tracking; r_ref
        is the reference yaw RATE (heading-follow slew FF) used the same way in yaw.
        yaw_target (final edge heading) is accepted for interface parity with the
        DOB-MPC horizon yaw preview; the PID needs only the instantaneous r_ref FF."""
        if p_ref is not None: 
            self.p_ref = np.asarray(p_ref, float) # x,y,z by JJ
        if yaw_ref is not None:
            self.yaw_ref = float(yaw_ref) # only yaw because pitch and roll are passively stabilized by buoyancy by JJ
        if v_ref is not None:
            self.v_ref = np.asarray(v_ref, float) # vx, vy, vz by JJ
        if r_ref is not None:
            self.r_ref = float(r_ref) #wx, wy, wz by JJ

    @staticmethod
    def _yaw_from_R(R):
        return float(np.arctan2(R[1, 0], R[0, 0]))

    @staticmethod
    def _pitch_from_R(R):
        return float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))

    # ---- control law -----------------------------------------------------
    def compute(self, data):
        g = self.g
        p = np.asarray(data.xpos[self.bid], float)            # world position
        R = np.asarray(data.xmat[self.bid], float).reshape(3, 3)   # body->world
        v = np.asarray(data.qvel[:3], float)                  # world linear vel
        r = float(data.qvel[5])                               # body yaw rate
        yaw = self._yaw_from_R(R)
        pitch = self._pitch_from_R(R)

        kp = np.asarray(g["kp"]); kd = np.asarray(g["kd"]); ki = np.asarray(g["ki"])
        i_max = np.asarray(g["i_max"])
        e = self.p_ref - p                                    # world position error



        ########
        # i term in PID : reduce steady-state error by JJ
        """
        if we compund error in the begining, then when ROV is near the setpoint, the integral term will be too large and cause overshoot by JJ 
        so we only integrate when the error is small (within a gate) to avoid overshoot and windup by JJ
        """
        if self.use_i:
            gate = np.abs(e) < g["e_gate"] # True or False by JJ
            self._I = self._I + np.where(gate, e * self.dt, 0.0) # integrate only when |error| < g["e_gate"] by JJ
            i_cap = i_max / np.maximum(ki, 1e-9) 
            self._I = np.clip(self._I, -i_cap, i_cap) # clip the integral term to avoid windup by JJ
            i_term = ki * self._I # ki x compound error by JJ
        else:
            i_term = np.zeros(3)
        ######## 

        F_world = kp * e - kd * (v - self.v_ref) + i_term     # calculate the world force by JJ
        F_world[2] += -self.net_buoy                          # buoyancy compensation (world +z) by JJ
        F_body = R.T @ F_world                                # coordinate transformation by JJ



        # surge (body x): slew-rate limit(N/s) -> optional pitch guard -> amplitude clamp
        fx = F_body[0]
        dmax = g["surge_slew"] * self.dt # maximum Force by JJ
        fx = np.clip(fx, self._fx_prev - dmax, self._fx_prev + dmax) # clip the surge force to avoid sudden change by JJ
        
        # reduce surge when pitch is too large by JJ 
        guard = np.radians(g["pitch_guard_deg"]) # soft limit 
        if abs(pitch) > guard: 
            fx *= max(0.0, 1.0 - (abs(pitch) - guard) / np.radians(40.0)) 
        
        fx = float(np.clip(fx, -g["f_max"][0], g["f_max"][0])) # final clip to avoid too large surge force by JJ

        self._fx_prev = fx 
        F_body[0] = fx
        F_body[1] = float(np.clip(F_body[1], -g["f_max"][1], g["f_max"][1])) # clip sway force to avoid too large sway force by JJ
        F_body[2] = float(np.clip(F_body[2], -g["f_max"][2], g["f_max"][2])) # clip heave force to avoid too large heave force by JJ

        # yaw PD (+ gated integral)
        e_yaw = wrap_angle(self.yaw_ref - yaw)
        if self.use_i and abs(e_yaw) < g["yaw_gate"]:
            cap = g["yaw_i_max"] / max(g["yaw_ki"], 1e-9)
            self._I_yaw = float(np.clip(self._I_yaw + e_yaw * self.dt, -cap, cap))
        mz = g["yaw_kp"] * e_yaw - g["yaw_kd"] * (r - self.r_ref)
        if self.use_i:
            mz += g["yaw_ki"] * self._I_yaw
        mz = float(np.clip(mz, -g["mz_max"], g["mz_max"]))

        # OPTIONAL roll/pitch leveling PD (rank-6 only; gains provide rp_kp to enable).
        # Needed by heavy_gripper: the asymmetric payload (forward COM + jaw weight)
        # applies a static attitude torque comparable to the passive coBM restoring
        # (~1.1 N·m/rad), which alone lets the attitude walk off and eventually flip.
        # heavy/bluerov2 gains omit rp_kp -> Mx=My=0, byte-identical behavior.
        mx = my = 0.0
        if g.get("rp_kp") is not None:
            rp_kp = np.asarray(g["rp_kp"]); rp_kd = np.asarray(g["rp_kd"])
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
            p_rate, q_rate = float(data.qvel[3]), float(data.qvel[4])
            rp_max = float(g.get("rp_max", 8.0))
            mx = float(np.clip(-rp_kp[0] * roll - rp_kd[0] * p_rate, -rp_max, rp_max))
            my = float(np.clip(-rp_kp[1] * pitch - rp_kd[1] * q_rate, -rp_max, rp_max))

        wrench = np.array([F_body[0], F_body[1], F_body[2], mx, my, mz])
        self.commanded = wrench
        return wrench

    def apply(self, model, data):
        """Compute the wrench and write thruster forces. Returns (forces, realized)."""
        wrench = self.compute(data)
        forces, realized = T.set_wrench_command(model, data, wrench, self.B,
                                                actuator=self.actuator)
        self.realized = np.asarray(realized, float)
        return forces, self.realized

    # ---- convenience for tests / logging ---------------------------------
    def pos_error(self, data):
        return float(np.linalg.norm(self.p_ref - np.asarray(data.xpos[self.bid], float)))

    def yaw_error(self, data):
        R = np.asarray(data.xmat[self.bid], float).reshape(3, 3)
        return float(wrap_angle(self.yaw_ref - self._yaw_from_R(R)))
