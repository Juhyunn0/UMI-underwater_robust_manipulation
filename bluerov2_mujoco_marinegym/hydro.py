#!/usr/bin/env python3
"""
Fossen-style underwater dynamics for the MarineGym BlueROV2 (Phase 3), FLU frame.

Adds buoyancy + restoring moment, added mass, and linear+quadratic drag, using
MarineGym's coefficients from marinegym_assets/BlueROV.yaml. Installed as a
MuJoCo passive-force callback (set_mjcb_passive) so it runs every substep inside
plain mj_step AND inside the managed viewer (teleop) with no per-step Python loop.

What is applied each substep (about the COM; COM == body origin here):
  * Buoyancy B = rho*g*V upward (+z world) at the CB = COM + coBM*z_body. Because
    the CB is coBM above the COM, a tilt produces a RESTORING moment that rights
    the vehicle (this passively stabilizes the Phase-2 underactuated pitch).
    Gravity (m*g down at COM) is applied by MuJoCo itself.
  * Drag (Fossen D(v)v):  -(D_L + D_NL*|nu|) * nu           [linear + quadratic]
  * Added-mass Coriolis:  -C_A(nu) nu   (Fossen, diagonal M_A)
  * Added-mass inertial:  -M_A * nudot  (nudot = one-substep-lagged, low-pass
    filtered body-frame acceleration -- the uuv_simulator / MarineGym technique;
    stable at dt=2 ms even though heave added mass 14.57 > body mass 11.2).

Frame: FLU (x fwd, y left, z up), gravity (0,0,-9.81). No NED. Coefficients are
MarineGym's only. Only mujoco + numpy required (a tiny YAML reader avoids pyyaml).

Implementation choice for added mass (documented per the task): translational AND
rotational added mass are applied as explicit lagged/filtered -M_A*nudot forces
(exactly MarineGym's method), rather than folding the rotational part into the XML
inertia. This keeps bluerov.xml the pure rigid body and reproduces MarineGym
term-for-term; the 0.3 low-pass filter is what makes the >mass added-mass stable.
"""
import os
import numpy as np
import mujoco

import rov_model as RM     # which variant's yaml (BlueROV vs BlueROVHeavy); env ROV_MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
# default coeffs track the selected ROV variant (heavy only differs in buoyant
# volume; added mass / drag are identical). Override per-call via coeff_path.
YAML = RM.YAML_PATH
RHO_FRESHWATER = 997.0   # kg/m^3 — MarineGym calculate_buoyancy uses 997


def load_coeffs(path=YAML):
    """Minimal reader for MarineGym's flat BlueROV.yaml (scalars + simple lists)."""
    scal, vecs, cur = {}, {}, None
    for raw in open(path).read().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("- "):
            if cur is not None:
                vecs.setdefault(cur, []).append(float(s[2:].strip()))
            continue
        if ":" in s:
            k, _, v = s.partition(":")
            k, v = k.strip(), v.strip()
            if v == "":
                cur = k                     # a list / nested block follows
            else:
                scal[k] = v
                cur = None
    return dict(
        volume=float(scal["volume"]),
        coBM=float(scal["coBM"]),
        drag_coef=float(scal["drag_coef"]),
        added_mass=np.array(vecs["added_mass"]),
        linear_damping=np.array(vecs["linear_damping"]),
        quadratic_damping=np.array(vecs["quadratic_damping"]),
    )


class Hydrodynamics:
    def __init__(self, model, coeff_path=YAML, rho=RHO_FRESHWATER,
                 acc_filter=0.3, buoyancy=None, body="base_link", disturbance=None,
                 diag_wtrue=False):
        self.model = model
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        # Phase 4: optional environmental disturbance field (current/waves/kicks).
        # If set, hydro uses the RELATIVE velocity vr = v - v_water in drag/
        # Coriolis/added-mass (so the flow carries an unpowered vehicle) and adds
        # the kick external force. None -> still water (Phase-3 behaviour).
        self.disturbance = disturbance
        # Read-only EAOB ground-truth diagnostic (does NOT change the applied forces):
        # when enabled, also evaluate the still-water (nu) wrench and store the lumped
        # disturbance w_true_world = (plant force - still-water model force) + FK ext,
        # in FLU world, for comparison vs DOBMPCController.w_world_flu(). Off by default.
        self.diag_wtrue = bool(diag_wtrue)
        self.w_true_world = np.zeros(6)
        c = load_coeffs(coeff_path)
        self.volume = c["volume"]
        self.coBM = c["coBM"]
        self.M_A = c["added_mass"]            # [Xu', Yv', Zw', Kp', Mq', Nr']
        self.D_L = c["linear_damping"]
        self.D_NL = c["quadratic_damping"]
        self.rho = rho
        self.g = abs(float(model.opt.gravity[2]))   # 9.81, consistent for both
        self.mass = float(model.body_mass[self.bid])
        self.weight = self.mass * self.g
        self.buoyancy = self.rho * self.g * self.volume if buoyancy is None else buoyancy
        self.acc_filter = acc_filter
        self.dt = float(model.opt.timestep)
        self._last_drag = np.zeros(6)
        self._last_added = np.zeros(6)
        self.components = {}     # name -> (world point, world force vector) for viz
        self.water = {}          # name -> (world point, world velocity vector) for viz
        self.reset()

    def reset(self):
        self._nu_prev = None
        self._nudot_f = np.zeros(6)
        # parallel filter state for the read-only still-water diagnostic (diag_wtrue)
        self._nu_prev_still = None
        self._nudot_f_still = np.zeros(6)
        self._wb_still = np.zeros(6)

    # body-frame velocity nu = [v(3); omega(3)] at the COM (== body origin)
    def nu(self, data):
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.bid, res, 1)            # 1 = local frame
        return np.concatenate([res[3:6], res[0:3]])           # [lin; ang]

    def _coriolis_added(self, nu):
        """C_A(nu) nu for diagonal added mass M_A (Fossen)."""
        a = self.M_A
        u, v, w, p, q, r = nu
        return np.array([
            a[2]*w*q - a[1]*v*r,
            -a[2]*w*p + a[0]*u*r,
            a[1]*v*p - a[0]*u*q,
            a[2]*w*v - a[1]*v*w + a[5]*r*q - a[4]*q*r,
            -a[2]*w*u + a[0]*u*w - a[5]*r*p + a[3]*p*r,
            a[1]*v*u - a[0]*u*v + a[4]*q*p - a[3]*p*q,
        ])

# Hydrodynamic Model 
    def wrench_body(self, data):
        """Drag + added-mass(+Coriolis) wrench in the body frame (excl. buoyancy).

        Uses the RELATIVE velocity vr = v - v_water when a disturbance field is
        attached. v_water (current + waves, FLU world) is rotated into the body
        frame; its angular part is zero. Because vr also drives the added-mass
        finite difference, a time-varying v_water (waves) excites both the drag
        and the added-mass (inertia) force -- Morison-like -- with no extra term.
        """
        nu = self.nu(data)
        nu_r = nu.copy()
        if self.disturbance is not None and self.disturbance.enabled:
            R = data.xmat[self.bid].reshape(3, 3)
            v_water = self.disturbance.water_velocity(data.time, data.xipos[self.bid])
            nu_r[:3] = nu[:3] - R.T @ v_water        # relative linear vel(water doesn't have angular part), body frame (by JJ)


        # this is for the added-mass finite-difference (nudot) term; the filter is what makes it stable (by JJ)
        if self._nu_prev is None:
            nudot = np.zeros(6)
        else:
            nudot = (nu_r - self._nu_prev) / self.dt
        self._nudot_f = self.acc_filter * nudot + (1 - self.acc_filter) * self._nudot_f # acts as low-pass filter on the added-mass force (by JJ)
        self._nu_prev = nu_r

        # For the forces that MuJoCo doesn't know about; drag(body) + added mass effect + Coriolis (added mass) (by JJ) 
        f_drag = -(self.D_L * nu_r + self.D_NL * np.abs(nu_r) * nu_r)  # opposes rel. motion  # linear drag + quadratic drag (by JJ)
        f_added = -self.M_A * self._nudot_f                            # opposes rel. accel   # added-mass inertia multiplied by low-pass filter (by JJ)
        f_cor = -self._coriolis_added(nu_r)                                                   # Coriolis (added mass) (by JJ)

        self._last_drag = f_drag      # stored (read-only) for visualization
        self._last_added = f_added
        if self.diag_wtrue:
            self._update_wb_still(nu)


        return f_drag + f_added + f_cor

    def _update_wb_still(self, nu):
        """Read-only diagnostic: the hydro wrench the controller's STILL-WATER internal
        model expects (drag+added+Coriolis at the absolute velocity nu, no flow), with
        its OWN added-mass filter state. self._wb_still is subtracted from the realised
        (nu_r) wrench to form the lumped disturbance the EAOB estimates. No force change."""
        if self._nu_prev_still is None:
            nudot = np.zeros(6)
        else:
            nudot = (nu - self._nu_prev_still) / self.dt
        self._nudot_f_still = (self.acc_filter * nudot
                               + (1 - self.acc_filter) * self._nudot_f_still)
        self._nu_prev_still = nu
        f_drag = -(self.D_L * nu + self.D_NL * np.abs(nu) * nu)
        f_added = -self.M_A * self._nudot_f_still
        f_cor = -self._coriolis_added(nu)
        self._wb_still = f_drag + f_added + f_cor

    def __call__(self, model, data):
        """Passive-force callback: add the hydro wrench to qfrc_passive."""
        bid = self.bid
        R = data.xmat[bid].reshape(3, 3)
        com = data.xipos[bid]

        # buoyancy (world +z) applied at the CB = COM + coBM along body +z;
        # mj_applyFT turns the offset point into the restoring moment for us.
        cb = com + R @ np.array([0.0, 0.0, self.coBM])  # self.coBM is the distance from COM to CB along body +z (by JJ)

        # 1. Buoyancy (world +z) at the CB (COM + coBM along body +z) (by JJ)
        mujoco.mj_applyFT(model, data, np.array([0.0, 0.0, self.buoyancy]),
                          np.zeros(3), cb, bid, data.qfrc_passive)

        # 2. drag + added mass (body frame) -> world, applied at the COM (by JJ)
        wb = self.wrench_body(data)
        mujoco.mj_applyFT(model, data, R @ wb[:3], R @ wb[3:], com, bid,
                          data.qfrc_passive)
        
        # 3. external force at the COM (by JJ) 
        # Phase 4: impulsive kick (world-frame external force) at the COM
        Fk = np.zeros(3)
        if self.disturbance is not None and self.disturbance.enabled:
            Fk, Tk = self.disturbance.external_wrench(data.time, com)
            if Fk.any() or Tk.any():
                mujoco.mj_applyFT(model, data, Fk, Tk, com, bid, data.qfrc_passive)

        # read-only lumped-disturbance ground truth (FLU world): drag/added/cor
        # difference (nu_r vs still-water nu) rotated to world + the FK external force.
        if self.diag_wtrue:
            dwb = wb - self._wb_still
            self.w_true_world = np.concatenate([R @ dwb[:3] + Fk, R @ dwb[3:]])

        # ---- expose per-component forces for visualization (READ-ONLY: this
        #      records what was already applied; it does not change dynamics) ----
        self.components = {
            "buoyancy": (cb.copy(), np.array([0.0, 0.0, self.buoyancy])),  # world, at CB
            "drag":  (com.copy(), R @ self._last_drag[:3]),                # world, at COM
            "added": (com.copy(), R @ self._last_added[:3]),
            "kick":  (com.copy(), Fk.copy()),
        }
        if self.disturbance is not None:
            self.water = {
                "current": (com.copy(), self.disturbance.current_velocity()),
                "wave":    (com.copy(), self.disturbance.wave_velocity(data.time, com)),
            }
        else:
            self.water = {"current": (com.copy(), np.zeros(3)),
                          "wave": (com.copy(), np.zeros(3))}

    def install(self):
        mujoco.set_mjcb_passive(self)
        return self

    @staticmethod
    def uninstall():
        mujoco.set_mjcb_passive(None)

    def summary(self):
        net = self.buoyancy - self.weight
        return (f"hydro: mass={self.mass:.3f} kg  weight={self.weight:.2f} N  "
                f"buoyancy={self.buoyancy:.2f} N (rho={self.rho:.0f}, V={self.volume:.6f} m^3)\n"
                f"   net buoyancy = {net:+.2f} N ({'slightly positive' if net>0 else 'negative'}), "
                f"coBM = {self.coBM:.3f} m (CB above COM)\n"
                f"   added_mass        M_A  = {self.M_A.tolist()}\n"
                f"   linear_damping    D_L  = {self.D_L.tolist()}\n"
                f"   quadratic_damping D_NL = {self.D_NL.tolist()}")
