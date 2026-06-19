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


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# Starting gains (tune from here). Translational gains are WORLD-frame.
DEFAULT_GAINS = dict(
    kp=(6.0, 12.0, 20.0),       # surge, sway, heave  [N/m]
    kd=(8.0, 12.0, 22.0),       #                     [N·s/m]  (heave Kd=22 -> zeta~0.65)
    ki=(1.0, 1.5, 1.2),         #                     [N/(m·s)]
    i_max=(4.0, 5.0, 5.0),      # |Ki*integral| clamp [N]
    f_max=(6.0, 30.0, 30.0),    # surge/sway/heave force saturation [N]. Surge kept
                                # low: a steady Fx balances the weak pitch restoring
                                # (1.11 N·m) at sin(theta)=0.0725·Fx/1.11, so 6 N -> ~23°.
    surge_slew=30.0,            # surge force slew-rate limit [N/s] (softens pitch kick)
    yaw_kp=5.0, yaw_kd=3.0, yaw_ki=0.5, yaw_i_max=2.0, mz_max=10.0,  # yaw [N·m]
    e_gate=0.5, yaw_gate=0.2,   # integrate only when |error| < gate (m, rad)
    pitch_guard_deg=15.0,       # soft-scale surge above this |pitch| (deg)
)


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
        self.commanded = np.zeros(6)
        self.realized = np.zeros(6)
        if getattr(self, "actuator", None) is not None:
            self.actuator.reset()

    def set_target(self, p_ref=None, yaw_ref=None, v_ref=None):
        """Update the (possibly moving) setpoint. v_ref is the reference world velocity
        used as a feed-forward in the derivative term, for trajectory tracking."""
        if p_ref is not None:
            self.p_ref = np.asarray(p_ref, float)
        if yaw_ref is not None:
            self.yaw_ref = float(yaw_ref)
        if v_ref is not None:
            self.v_ref = np.asarray(v_ref, float)

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

        # gated anti-windup integral (only near the setpoint, then clamped)
        if self.use_i:
            gate = np.abs(e) < g["e_gate"]
            self._I = self._I + np.where(gate, e * self.dt, 0.0)
            i_cap = i_max / np.maximum(ki, 1e-9)
            self._I = np.clip(self._I, -i_cap, i_cap)
            i_term = ki * self._I
        else:
            i_term = np.zeros(3)

        F_world = kp * e - kd * (v - self.v_ref) + i_term     # v_ref = trajectory FF
        F_world[2] += -self.net_buoy                          # hold depth vs +buoyancy
        F_body = R.T @ F_world                                # rotate force to body

        # surge (body x): slew-rate limit -> optional pitch guard -> amplitude clamp
        fx = F_body[0]
        dmax = g["surge_slew"] * self.dt
        fx = np.clip(fx, self._fx_prev - dmax, self._fx_prev + dmax)
        guard = np.radians(g["pitch_guard_deg"])
        if abs(pitch) > guard:
            fx *= max(0.0, 1.0 - (abs(pitch) - guard) / np.radians(40.0))
        fx = float(np.clip(fx, -g["f_max"][0], g["f_max"][0]))
        self._fx_prev = fx
        F_body[0] = fx
        F_body[1] = float(np.clip(F_body[1], -g["f_max"][1], g["f_max"][1]))
        F_body[2] = float(np.clip(F_body[2], -g["f_max"][2], g["f_max"][2]))

        # yaw PD (+ gated integral)
        e_yaw = wrap_angle(self.yaw_ref - yaw)
        if self.use_i and abs(e_yaw) < g["yaw_gate"]:
            cap = g["yaw_i_max"] / max(g["yaw_ki"], 1e-9)
            self._I_yaw = float(np.clip(self._I_yaw + e_yaw * self.dt, -cap, cap))
        mz = g["yaw_kp"] * e_yaw - g["yaw_kd"] * r
        if self.use_i:
            mz += g["yaw_ki"] * self._I_yaw
        mz = float(np.clip(mz, -g["mz_max"], g["mz_max"]))

        wrench = np.array([F_body[0], F_body[1], F_body[2], 0.0, 0.0, mz])  # Mx=0,My=0
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
