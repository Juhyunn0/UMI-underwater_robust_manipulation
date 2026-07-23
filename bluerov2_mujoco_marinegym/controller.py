#!/usr/bin/env python3
"""Baseline PD/PID setpoint (go-to-origin) controller for the BlueROV2 Heavy (FLU).

Drives the vehicle to a world setpoint (default: global origin) + yaw reference in
the MarineGym FLU model. Fully FLU — no NED (that conversion is only the DOB-MPC's
boundary). The heavy family is rank-6 FULLY ACTUATED, so all 6 DOF are independently
commandable: surge/sway/heave + yaw, PLUS optional roll/pitch leveling (enabled per
variant by rp_kp; on for heavy_gripper, whose asymmetric payload needs it). Passive
buoyancy restoring still stabilizes roll/pitch when the leveling loop is off.

Design (reviewed by control-theory-advisor):
  * World-frame position PID; rotate ONLY the final force to body (avoids the
    rotating-stiffness / crabbing issue with anisotropic gains).
  * D term uses world linear velocity (data.qvel[:3]); yaw D uses body yaw rate.
  * Net-buoyancy feedforward, and GATED + CLAMPED anti-windup integrals on position,
    yaw, and (PID mode) roll/pitch leveling — integrate only near the setpoint.
  * Body forces saturated in WRENCH space + a uniform slew-rate limit across all three
    axes (per-thruster clipping would distort the commanded wrench direction).
Reuses thrusters.py allocation (allocation_matrix / set_wrench_command).
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
# ωn to ≤0.54), e_gate 0.15 m, slew 120 N/s, and the yaw rate reference
# feed-forward r_ref in compute(). Sim-only validity (ideal thrusters, perfect
# state, 500 Hz) — derate ωn to 1.0–1.5 and filter D for hardware.
GAINS_HEAVY = dict(
    kp=(131.6, 131.6, 141.8),   # surge, sway, heave  [N/m]
    kd=(90.6, 90.6, 99.1),      #                     [N·s/m]
    ki=(38.7, 38.7, 41.7),      #                     [N/(m·s)]
    i_max=(4.0, 5.0, 5.0),      # |Ki*integral| clamp [N]
    f_max=(30.0, 30.0, 30.0),   # wrench-space force saturation [N] (rank-6: no
                                # surge-pitch coupling, so no surge derate)
    slew=120.0,                 # uniform body-force slew-rate limit [N/s] (all axes)
    yaw_kp=8.95, yaw_kd=4.32, yaw_ki=3.95, yaw_i_max=2.0, mz_max=10.0,  # yaw [N·m]
    e_gate=0.15, yaw_gate=0.2,  # integrate only when |error| < gate (m, rad)
    pitch_guard_deg=15.0,       # dormant transient-pitch limit (dynamic surge->pitch)
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
    slew=120.0,                 # uniform body-force slew-rate limit [N/s] (all axes)
    yaw_kp=9.92, yaw_kd=4.79, yaw_ki=4.38, yaw_i_max=2.0, mz_max=10.0,  # yaw [N·m]
    e_gate=0.15, yaw_gate=0.2,  # integrate only when |error| < gate (m, rad)
    pitch_guard_deg=15.0,       # dormant backup limit (rp-PID leveling handles pitch first)
    # ACTIVE roll/pitch leveling (rank-6): the payload's static attitude torque
    # (forward COM + jaw weight, ~0.2-0.5 N·m) rivals the passive restoring
    # B·coBM ≈ 1.1 N·m/rad, so without this the attitude walks off and flips.
    # kp/kd from PD pole placement at ωn=3, ζ=0.9 on I_eff (roll 0.481, pitch 0.859).
    rp_kp=(4.3, 7.7), rp_kd=(2.6, 4.6), rp_max=8.0,   # roll, pitch [N·m]
    # PID leveling (2026-07-21): the integral nulls the steady tilt PD leaves under a
    # constant payload torque (φ_ss = τ/(kp+B_restore) ≈ 2-5° → ~cm end-effector error
    # at reach; any Ki>0 makes the loop type-1 → φ_ss→0). ki = I_eff·α·ωn³ (same pole
    # placement as translation, α=0.2, ωn=3): roll 0.481·5.4=2.60, pitch 0.859·5.4=4.64.
    # Gated (|angle|<rp_gate) + clamped (|ki·I|<rp_i_max) anti-windup; verified 3rd-order
    # poles all LHP with dominant ζ≈0.83-0.89 (integral does NOT erode PD damping).
    rp_ki=(2.60, 4.64), rp_i_max=(1.5, 2.0), rp_gate=0.15,   # roll, pitch
)

# Per-variant gains. Every variant is now rank-6 fully-actuated, so unknown variants
# (e.g. heavy_c3) fall back on GAINS_HEAVY.
_GAINS_BY_MODEL = {"heavy": GAINS_HEAVY, "heavy_gripper": GAINS_HEAVY_GRIPPER}
DEFAULT_GAINS = _GAINS_BY_MODEL.get(RM.MODEL, GAINS_HEAVY)


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
        self._I_rp = np.zeros(2)             # roll, pitch leveling integral (PID mode)
        self._f_prev = np.zeros(3)           # previous body force (uniform slew state)
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
        p = np.asarray(data.xpos[self.bid], float)            # world position (x,y,z)
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



        # Body-force shaping (rank-6 heavy: axis-symmetric over surge/sway/heave).
        # (1) a UNIFORM slew-rate limit models finite actuator/command bandwidth (abrupt
        #     wrench steps are unphysical -> smoother thruster demand, better sim2real);
        # (2) a per-axis amplitude clamp = finite thruster authority. Both act in WRENCH
        #     space so the commanded force DIRECTION is preserved (per-thruster clipping
        #     would distort it). by JJ
        f = np.array(F_body[:3], float)                          # working copy of body force
        f_max = np.asarray(g["f_max"], float)
        dmax = g["slew"] * self.dt                               # max Δforce per step
        f = np.clip(f, self._f_prev - dmax, self._f_prev + dmax) # slew-rate limit
        # Soft pitch guard: scale surge down when |pitch| is large. Kept for the heavy
        # family (gated on the pitch_guard_deg gain key). NOTE — although the rank-6
        # allocation STATICALLY decouples Fx from My (pure Fx -> My=0), a DYNAMIC
        # Coriolis/added-mass coupling still lets fast surge excite the weakly-restored
        # pitch mode: an aggressive far-offset slew transiently swings pitch to ~70°
        # WITHOUT this guard (test_controller) vs <45° with it. So it is useful on heavy
        # too, as a dormant transient-pitch limit (fires only above pitch_guard_deg, never
        # in normal small-error operation; on heavy_gripper the rp-PID leveling handles
        # attitude first and this only backs it up on large excursions). by JJ
        if "pitch_guard_deg" in g:
            guard = np.radians(g["pitch_guard_deg"])
            if abs(pitch) > guard:
                f[0] *= max(0.0, 1.0 - (abs(pitch) - guard) / np.radians(40.0))
        f = np.clip(f, -f_max, f_max)                            # amplitude clamp
        self._f_prev = f
        F_body[:3] = f

        # yaw PID (+ gated integral) # same algorithm as position PID by JJ
        e_yaw = wrap_angle(self.yaw_ref - yaw)
        if self.use_i and abs(e_yaw) < g["yaw_gate"]:
            cap = g["yaw_i_max"] / max(g["yaw_ki"], 1e-9)
            self._I_yaw = float(np.clip(self._I_yaw + e_yaw * self.dt, -cap, cap))
        mz = g["yaw_kp"] * e_yaw - g["yaw_kd"] * (r - self.r_ref)
        if self.use_i:
            mz += g["yaw_ki"] * self._I_yaw
        mz = float(np.clip(mz, -g["mz_max"], g["mz_max"]))

        # OPTIONAL roll/pitch leveling PID (rank-6 only; gains provide rp_kp to enable).
        # Needed by heavy_gripper: the asymmetric payload (forward COM + jaw weight)
        # applies a static attitude torque comparable to the passive coBM restoring
        # (~1.1 N·m/rad), which alone lets the attitude walk off and eventually flip.
        # The GATED + CLAMPED integral (PID mode, gains provide rp_ki) nulls the steady
        # tilt PD leaves under that constant torque: PD gives φ_ss = τ/(kp+B_restore) ≠ 0,
        # but any Ki>0 makes the loop type-1 so φ_ss -> 0 (matters for manipulation
        # end-effector accuracy). Anti-windup mirrors the position/yaw loops: integrate
        # -angle only when near level (|angle| < rp_gate) and clamp |ki·I| < rp_i_max.
        # heavy/bluerov2 gains omit rp_kp -> Mx=My=0; an rp_kp-only variant (or mode="pd")
        # stays pure PD, byte-identical to before.
        mx = my = 0.0
        if g.get("rp_kp") is not None:
            rp_kp = np.asarray(g["rp_kp"]); rp_kd = np.asarray(g["rp_kd"])
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
            p_rate, q_rate = float(data.qvel[3]), float(data.qvel[4])
            rp_max = float(g.get("rp_max", 8.0))
            if self.use_i and g.get("rp_ki") is not None:
                rp_ki = np.asarray(g["rp_ki"]); rp_i_max = np.asarray(g["rp_i_max"])
                ang = np.array([roll, pitch])                   # leveling error = -ang
                gate = np.abs(ang) < g.get("rp_gate", 0.15)     # integrate only near level
                self._I_rp = self._I_rp + np.where(gate, -ang * self.dt, 0.0)
                cap = rp_i_max / np.maximum(rp_ki, 1e-9)
                self._I_rp = np.clip(self._I_rp, -cap, cap)     # clamp so |ki·I| < rp_i_max
                i_rp = rp_ki * self._I_rp
            else:
                i_rp = np.zeros(2)
            mx = float(np.clip(-rp_kp[0] * roll - rp_kd[0] * p_rate + i_rp[0], -rp_max, rp_max))
            my = float(np.clip(-rp_kp[1] * pitch - rp_kd[1] * q_rate + i_rp[1], -rp_max, rp_max))

        wrench = np.array([F_body[0], F_body[1], F_body[2], mx, my, mz])
        self.commanded = wrench
        return wrench

    def apply(self, model, data):
        """Compute the wrench and write thruster forces. Returns (forces, realized)."""
        wrench = self.compute(data)
        forces, realized = T.set_wrench_command(model, data, wrench, self.B,
                                                actuator=self.actuator) # assign the wrench to the thrusters by JJ 
                                                # if we select actuator, then we will use the realistic thrusters to allocate the wrench to the thrusters by JJ
        self.realized = np.asarray(realized, float)
        return forces, self.realized 
        # force : [F1, F2, F3, F4, F5, F6, F7, F8], each motor force 
        # realized : [Fx, Fy, Fz, Mx, My, Mz], the actual wrench after allocation by JJ




    # ---- convenience for tests / logging ---------------------------------
    def pos_error(self, data):
        return float(np.linalg.norm(self.p_ref - np.asarray(data.xpos[self.bid], float)))

    def yaw_error(self, data):
        R = np.asarray(data.xmat[self.bid], float).reshape(3, 3)
        return float(wrap_angle(self.yaw_ref - self._yaw_from_R(R)))
