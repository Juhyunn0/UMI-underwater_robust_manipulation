#!/usr/bin/env python3
"""DOB-MPC controller for the marinegym BlueROV2 sim (FLU).

Wraps the validated NED EAOB + NMPC (dobmpc/ subpackage) as a drop-in alternative
to PoseController: same set_target / compute / apply interface, same rank-5
thrusters.py allocation, so teleop.py and the mission classes drive it unchanged.

Per control tick (20 Hz, ZOH between ticks over 25 physics substeps):
  1. read FLU state (mj_objectVelocity local=1 for body-frame nu -- NOT qvel[:3])
  2. frames.flu_to_ned -> [eta; nu] NED;  a_meas = d(nu)/dt by finite difference
     over the CONTROL tick (never data.qacc -- marinegym applies added mass as an
     external force, so qacc would double-count it in the EAOB measurement model)
  3. EAOB.update(meas, tau_cmd_ned)  (dobmpc mode) -> w_hat ; plain MPC -> w_hat=0
  4. NMPC.solve(x_ned, w_hat, xref_ned) -> u = [X, Y, Z, N]
  5. tau_ned = [X,Y,Z,0,0,N] -> frames.ned_wrench_to_flu -> tau_FLU (ZOH)
  6. thrusters.set_wrench_command (rank-5 pinv projects out the uncommanded pitch)

Design notes (control-theory-advisor validated): pitch is left to float to its
physical trim (MPC_Q pitch-weight 0, the EAOB absorbs the steady surge->pitch
coupling into w); the disturbance model is w_dot=0, so the DC current is rejected
strongly while the JONSWAP wave band / kicks are only partially rejected.
"""
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thrusters as T
from dobmpc import frames
from dobmpc import params as P
from dobmpc.fossen import rot_ib, wrap_angle
from dobmpc.eaob import EAOB
from dobmpc.mpc import make_nmpc


def _Rz_flu(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class DOBMPCController:
    def __init__(self, model, hydro=None, mode="dobmpc", setpoint=(0.0, 0.0, 0.0),
                 yaw_ref=0.0, body="base_link", ctrl_hz=20.0, N=P.MPC_N, actuator=None,
                 eaob_profile="perf", meas_noise=None, noise_seed=0,
                 mpc_state_source=None):
        """eaob_profile: EAOB tuning ("perf" = sensor-sigma covariances + NIS gate,
        "verify" = legacy near-deadbeat DT templates). meas_noise: corrupt the
        EAOB's eta/nu/nudot with Gaussian noise from params.EAOB_SIG_* (the same
        sigmas the perf filter R assumes); None -> on iff profile is "perf", so
        the consistent pairs (verify, clean) / (perf, noisy) are the defaults.
        mpc_state_source: what the NMPC consumes as x0 --
          "meas"     = the state-estimator output the loop actually has (in sim:
                       the clean reading + sensor noise, the SAME sample the EAOB
                       consumes; on hardware: the SLAM / AI-model estimator). The
                       DOB plug-in architecture -- the EAOB hands the MPC ONLY
                       w_hat, never its filtered eta_hat/nu_hat. Default under
                       noise, for BOTH mpc and dobmpc, so the two differ only by
                       w_hat (single-variable ablation).
          "estimate" = the EAOB-filtered eta_hat/nu_hat (offset-free-MPC style;
                       dobmpc only). The 2026-07-23 (2) default, now an opt-in.
          "truth"    = the clean plant readout (regression / observer isolation;
                       also the no-noise path).
        None -> params.MPC_STATE_SOURCE when measurement noise is active, else
        "truth". Whatever the source, it feeds ONLY the MPC x0 + its yaw-ref
        anchor -- measurement generation, the EAOB's inputs, and every
        truth-logging surface stay on the plant readout."""
        assert mode in ("dobmpc", "mpc"), mode
        assert eaob_profile in ("perf", "verify"), eaob_profile
        self.meas_noise = (eaob_profile == "perf") if meas_noise is None else bool(meas_noise)
        if mpc_state_source is None:
            src = P.MPC_STATE_SOURCE if self.meas_noise else "truth"
        else:
            src = mpc_state_source
        assert src in ("meas", "estimate", "truth"), src
        assert not (src == "estimate" and mode != "dobmpc"), \
            "estimate state source needs the EAOB -- use mode='dobmpc'"
        self.mpc_state_source = src
        self.model = model
        self.actuator = actuator                 # optional realistic thrusters (opt-in)
        self.hydro = hydro                       # only for parity with PoseController
        self.mode = mode
        self.eaob_profile = eaob_profile         # self.meas_noise set above (x0 default)
        self.noise_seed = int(noise_seed)
        self._noise_rng = np.random.default_rng(self.noise_seed)
        self._sig_eta = np.concatenate([P.EAOB_SIG_POS, P.EAOB_SIG_ANG])
        self._sig_nu = np.concatenate([P.EAOB_SIG_LVEL, P.EAOB_SIG_AVEL])
        self._sig_nudot = np.concatenate([np.full(3, P.EAOB_SIG_ACC),
                                          np.full(3, P.EAOB_SIG_AACC)])
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.B, _ = T.allocation_matrix(model)   # constant rank-5 body geometry

        self.dt_sim = float(model.opt.timestep)
        self.ctrl_dt = 1.0 / float(ctrl_hz)
        self.decim = max(1, round(self.ctrl_dt / self.dt_sim))   # substeps per tick
        self.ctrl_dt = self.decim * self.dt_sim                  # exact hold duration
        assert abs(self.ctrl_dt - P.DT_CTRL) < 1e-9, (
            f"control dt {self.ctrl_dt} != params DT_CTRL {P.DT_CTRL}; EAOB/MPC assume "
            f"DT_CTRL -- keep them equal")

        # references (world FLU)
        self.p_ref = np.asarray(setpoint, float)
        self.yaw_ref = float(yaw_ref)
        self.v_ref = np.zeros(3)                 # world-FLU reference velocity (DP: 0)
        self.r_ref = 0.0                         # reference yaw rate (heading-follow FF)
        self.yaw_target = float(yaw_ref)         # final edge heading for the yaw preview
        self._ref_traj = None                    # mission trajectory sampler (tracking)

        self.nmpc = make_nmpc(N=N, dt=P.DT_CTRL)   # acados (fast) or ipopt (ref)
        self.eaob = None                         # lazy-init at first tick (needs eta0)
        self._tau_flu = np.zeros(6)              # ZOH wrench held between ticks
        self._tau_ned_cmd = np.zeros(6)          # commanded NED wrench (fed to EAOB)
        self._nu_prev_ned = None
        self._k = 0                              # substep counter
        self._psi_ned_now = 0.0                  # current NED yaw, refreshed each tick
                                                 # (used to unwrap the yaw reference)

        self.commanded = np.zeros(6)
        self.realized = np.zeros(6)
        self.w_hat = np.zeros(6)                 # latest EAOB disturbance (NED body)
        self.solve_ms = 0.0
        self.n_fail = 0

    # ----------------------------------------------------- interface parity
    def set_target(self, p_ref=None, yaw_ref=None, v_ref=None, r_ref=None,
                   yaw_target=None):
        if p_ref is not None:
            self.p_ref = np.asarray(p_ref, float)
        if yaw_ref is not None:
            self.yaw_ref = float(yaw_ref)
        if v_ref is not None:
            self.v_ref = np.asarray(v_ref, float)
        if r_ref is not None:
            self.r_ref = float(r_ref)              # reference yaw rate (heading-follow FF)
        # final edge heading for the horizon yaw preview in _xref_ned; defaults to the
        # current yaw command so a caller that omits it -- or a straight leg -- gets no
        # preview (delta == 0). The NMPC uses yaw_ref + r_ref + yaw_target on turns.
        self.yaw_target = float(yaw_target) if yaw_target is not None else self.yaw_ref

    def set_reference_traj(self, fn):
        """Give the NMPC the mission's TIME-PARAMETERIZED reference (tracking mode).

        `fn(ts)` takes a 1-D array of absolute sim times (K,) and returns the
        reference at those times, all in world FLU (same conventions as set_target):
            p   (3, K)  position          yaw (K,)  heading command
            v   (3, K)  path velocity     r   (K,)  heading slew rate
        With a sampler set, every control tick fills the acados stage references
        yref_k from fn(t + k*dt), k = 0..N -- the standard receding-horizon
        reference preview of trajectory-tracking MPC -- so corners (position,
        heading AND velocity direction changes) inside the 3 s horizon are seen
        in advance instead of being straight-line-extrapolated through.
        `fn=None` clears it and falls back to the set_target interface, which
        remains the DP / teleop / station-keeping path (point stabilization and
        trajectory tracking are deliberately separate reference modes). reset()
        also clears the sampler -- re-arm it after reset, as run_compare/run_viewer
        do."""
        self._ref_traj = fn

    def reset(self):
        self.eaob = None
        self._noise_rng = np.random.default_rng(self.noise_seed)   # reproducible runs
        self.r_ref = 0.0
        self._ref_traj = None                    # tracking sampler does not survive reset
        self._tau_flu = np.zeros(6)
        self._tau_ned_cmd = np.zeros(6)
        self._nu_prev_ned = None
        self._k = 0
        self.nmpc.reset()
        if self.actuator is not None:
            self.actuator.reset()

    # ------------------------------------------------------------- state I/O
    def _read_state(self, data):
        p = np.asarray(data.xpos[self.bid], float)            # world FLU position
        R = np.asarray(data.xmat[self.bid], float).reshape(3, 3)
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.bid, res, 1)             # 1 = local (body) frame
        nu_flu = np.concatenate([res[3:6], res[0:3]])         # [lin; ang], body frame
        return p, R, nu_flu

    def _xref_ned(self, t=None):
        """Horizon trajectory reference (12, N+1) in NED.

        Two reference modes, dispatched on set_reference_traj:
          * TRACKING (sampler set, t given): delegate to _xref_ned_traj -- true
            receding-horizon preview sampled from the mission trajectory.
          * DP / SETPOINT (no sampler; this body): the set_target interface --
            constant pose extrapolated with the current v_ref/r_ref.

        Over the original constant-pose DP tile this body adds, all as RUNTIME
        reference data (model + cost untouched -> NO acados rebuild):
          * velocity feed-forward: the path velocity self.v_ref (world FLU) becomes
            the reference body velocity nu_ref (so the MPC stops fighting motion);
          * position preview: the setpoint is extrapolated forward over the horizon
            p_ref + v_ref*(k*dt), so position and velocity references are consistent;
          * yaw unwrap: the reference yaw is expressed relative to the current
            measured yaw (psi_now + wrap(psi_ref - psi_now)) so the NMPC cost -- which
            has no angle wrapping -- always turns the short way across +-pi.
          * yaw preview + yaw-rate FF (heading-follow turns): the yaw reference ramps
            over the horizon at r_ref toward the final edge heading yaw_target and is
            clamped there, so the NMPC anticipates the ongoing rotation instead of
            chasing a frozen target (see the block below).
        v_ref = 0 AND r_ref = 0 reduces it EXACTLY to the old constant-pose tile (the
        unwrap is a no-op when |psi_ref - psi_now| < pi), so DP / station-keeping is
        unchanged.
        """
        if getattr(self, "_ref_traj", None) is not None and t is not None:
            return self._xref_ned_traj(float(t))
        N = self.nmpc.N
        dt = P.DT_CTRL
        R_ref = _Rz_flu(self.yaw_ref)
        # orientation (constant over horizon): NED euler with yaw unwrapped vs now
        eta0 = frames.flu_to_ned_eta(self.p_ref, R_ref)
        eta0[5] = self._psi_ned_now + wrap_angle(eta0[5] - self._psi_ned_now)
        # velocity feed-forward: world-FLU v_ref -> reference body-FLU lin vel -> FRD
        nu_ned = np.concatenate([frames.S @ (R_ref.T @ self.v_ref), np.zeros(3)])
        # position preview: p_ref + v_ref*(k*dt) (world FLU) -> NED via S
        ks = np.arange(N + 1)
        pos_world = self.p_ref[:, None] + np.outer(self.v_ref, ks * dt)   # (3, N+1)
        xref = np.zeros((12, N + 1))
        xref[0:3, :] = frames.S @ pos_world
        xref[3:6, :] = eta0[3:6][:, None]
        xref[6:12, :] = nu_ned[:, None]

        # yaw preview + yaw-rate feed-forward (heading-follow TURNS only). During a
        # corner the yaw command slews at r_ref (world-FLU, ~+1.047 rad/s); a frozen
        # yaw target makes the NMPC lag and overshoot the turn. Extrapolate the future
        # command yaw over the horizon and CLAMP it at the final edge heading so it
        # never predicts past the corner. r_ref == 0 (straights / DP) -> exact no-op.
        if self.r_ref != 0.0:
            r_ned = -self.r_ref                     # world +yaw-rate -> NED/FRD r (S sign flip)
            psi0 = eta0[5]                          # current unwrapped NED command yaw
            eta_t = frames.flu_to_ned_eta(self.p_ref, _Rz_flu(self.yaw_target))
            psi_t = psi0 + wrap_angle(eta_t[5] - psi0)   # NED target, unwrapped vs psi0
            delta = psi_t - psi0                    # remaining rotation (small, unwrapped)
            step = np.clip(r_ned * ks * dt, min(0.0, delta), max(0.0, delta))
            xref[5, :] = psi0 + step                # yaw-angle preview  (Q weight 150)
            xref[11, :] = np.where(np.abs(step) < abs(delta) - 1e-12, r_ned, 0.0)  # rate FF (Q 10)
        return xref

    def _xref_ned_traj(self, t0):
        """Horizon reference (12, N+1) sampled from the mission trajectory.

        The standard trajectory-tracking form: stage k gets the TRUE reference at
        t0 + k*dt (position, heading, path velocity, heading rate), so a corner
        inside the horizon bends the position preview, rotates the stage-wise
        body-velocity reference onto the new leg, and ramps the yaw reference --
        the NMPC starts turning BEFORE the vertex instead of extrapolating
        straight through it (the _xref_ned setpoint body's known artifact).

        Frame handling mirrors _xref_ned exactly: world-FLU pose -> NED via
        frames.flu_to_ned_eta; per-stage body-FLU linear velocity R_k^T v_k -> FRD
        via S; world +yaw-rate -> NED r with the S sign flip. Yaw is unwrapped
        stage-to-stage anchored at the current measured yaw (_psi_ned_now), so the
        no-wrap NMPC cost always turns the short way across +-pi and stays
        continuous along the horizon."""
        N = self.nmpc.N
        dt = P.DT_CTRL
        ts = t0 + np.arange(N + 1) * dt
        p_w, yaw_w, v_w, r_w = self._ref_traj(ts)          # world FLU, vectorized
        p_w = np.asarray(p_w, float)
        v_w = np.asarray(v_w, float)
        yaw_w = np.asarray(yaw_w, float).ravel()
        r_w = np.asarray(r_w, float).ravel()
        # hard shape gate: a transposed (K,3) return would silently scramble
        # components under reshape -- fail loudly instead
        assert p_w.shape == (3, N + 1) and v_w.shape == (3, N + 1), \
            f"sampler must return (3,{N + 1}) p/v, got {p_w.shape}/{v_w.shape}"
        assert yaw_w.size == N + 1 and r_w.size == N + 1, \
            f"sampler must return {N + 1} yaw/r samples, got {yaw_w.size}/{r_w.size}"

        xref = np.zeros((12, N + 1))
        xref[0:3, :] = frames.S @ p_w
        psi_prev = self._psi_ned_now
        for k in range(N + 1):
            Rk = _Rz_flu(yaw_w[k])
            eta_k = frames.flu_to_ned_eta(p_w[:, k], Rk)
            psi_prev = psi_prev + wrap_angle(eta_k[5] - psi_prev)   # short-way unwrap
            xref[3:5, k] = eta_k[3:5]
            xref[5, k] = psi_prev
            xref[6:9, k] = frames.S @ (Rk.T @ v_w[:, k])   # body-FLU lin vel -> FRD
        xref[11, :] = -r_w                                  # world +yaw-rate -> NED r
        return xref

    # --------------------------------------------------------- control tick
    def _control_step(self, data):
        p, R, nu_flu = self._read_state(data)     # clean plant truth
        eta_ned = frames.flu_to_ned_eta(p, R)
        nu_ned = frames.flu_to_ned_nu(nu_flu)

        if self.eaob is None:                     # lazy init at the (clean) pose
            self.eaob = EAOB(eta0=eta_ned, nu0=nu_ned, profile=self.eaob_profile)
            self._nu_prev_ned = nu_ned.copy()

        a_meas = (nu_ned - self._nu_prev_ned) / self.ctrl_dt   # clean FD over the tick
        self._nu_prev_ned = nu_ned.copy()

        # ---- state-estimator output: ONE pose/vel sample per tick, feeding BOTH
        # the EAOB measurements AND the MPC x0 (when mpc_state_source=="meas"). In
        # sim = the clean reading + Gaussian sensor noise (params.EAOB_SIG_*: eta/
        # nu model vision+pressure/DVL, nudot an IMU accel + differentiated gyro,
        # so it gets its OWN noise on top of the clean FD, which -- with
        # _nu_prev_ned -- always tracks the true velocity); on hardware = the SLAM
        # / AI-model estimator. meas_noise off -> the estimator output IS truth.
        if self.meas_noise:
            rng = self._noise_rng
            eta_meas = eta_ned + rng.normal(0.0, self._sig_eta)
            nu_meas = nu_ned + rng.normal(0.0, self._sig_nu)
            nudot_meas = a_meas + rng.normal(0.0, self._sig_nudot)
        else:
            eta_meas, nu_meas, nudot_meas = eta_ned, nu_ned, a_meas

        eta_hat = nu_hat = None
        if self.mode == "dobmpc":
            eta_hat, nu_hat, self.w_hat = self.eaob.update(
                {"eta": eta_meas, "nu": nu_meas, "nudot": nudot_meas},
                self._tau_ned_cmd)
        else:                                     # plain MPC: no disturbance comp
            self.w_hat = np.zeros(6)

        # THE single x0 source switch (params.MPC_STATE_SOURCE):
        #   "meas"     -> the estimator output the loop has (DOB plug-in: the EAOB
        #                 feeds the MPC only w_hat, not its filtered state);
        #   "estimate" -> the EAOB-filtered state (offset-free-MPC style);
        #   "truth"    -> the clean plant state.
        # The yaw-ref anchor follows the SAME source so x0 yaw and the reference
        # sit on one branch (the NMPC cost has no angle wrapping; the reference
        # re-anchors to _psi_ned_now every tick; neither backend bounds yaw, so a
        # continuously accumulated estimate yaw is safe).
        if self.mpc_state_source == "estimate":
            eta0, nu0 = eta_hat, nu_hat
        elif self.mpc_state_source == "meas":
            eta0, nu0 = eta_meas, nu_meas
        else:                                     # "truth"
            eta0, nu0 = eta_ned, nu_ned
        self._psi_ned_now = float(eta0[5])
        u = self.nmpc.solve(np.concatenate([eta0, nu0]), self.w_hat,
                            self._xref_ned(data.time))
        self.n_fail = self.nmpc.n_fail
        if getattr(P, "FULLY_ACTUATED", False):
            # heavy: NU=6, the full wrench [X,Y,Z,K,M,N] is commanded and realized.
            tau_ned = np.array([u[0], u[1], u[2], u[3], u[4], u[5]])
            self._tau_ned_cmd = tau_ned                       # EAOB sees the full wrench
            self._tau_flu = frames.ned_wrench_to_flu(tau_ned)
        else:
            # bluerov2 (rank-5, option b): the EAOB is fed the commanded wrench INCLUDING
            # the modeled surge->pitch coupling (My = kappa*surge, NED), so it attributes
            # the realized pitch moment to control and keeps w[pitch] ~ 0 (no double-count
            # with the MPC model). The thruster command keeps My=0 -- the rank-5 allocation
            # realizes the coupling physically.
            kappa = P.SURGE_PITCH_COUPLING if getattr(P, "PITCH_AWARE", False) else 0.0
            self._tau_ned_cmd = np.array([u[0], u[1], u[2], 0.0, kappa * u[0], u[3]])
            self._tau_flu = frames.ned_wrench_to_flu(
                np.array([u[0], u[1], u[2], 0.0, 0.0, u[3]]))
        self.commanded = self._tau_flu.copy()

    # --------------------------------------------------------- public step
    def compute(self, data):
        """Return the body wrench currently held (parity with PoseController)."""
        return self._tau_flu.copy()

    def apply(self, model, data):
        """Run a control tick on the decimation boundary, hold (ZOH) otherwise,
        then allocate the held FLU wrench to thrusters. Returns (forces, realized)."""
        if self._k % self.decim == 0:
            self._control_step(data)
        self._k += 1
        forces, realized = T.set_wrench_command(model, data, self._tau_flu, self.B,
                                                actuator=self.actuator)
        self.realized = np.asarray(realized, float)
        return forces, self.realized

    # --------------------------------------------------- diagnostics
    def w_world_flu(self):
        """EAOB disturbance estimate in FLU world (for comparison vs the true
        current/wave, which live in FLU world). Zeros for plain MPC."""
        if self.eaob is None or self.mode != "dobmpc":
            return np.zeros(6)
        return frames.ned_w_world_to_flu(self.eaob.w_world())

    def eaob_meta(self):
        """Provenance dict for run meta sidecars (tuning + gate counters).
        Records recorded before this key existed are all verify/deadbeat +
        clean-measurement equivalent."""
        m = dict(profile=self.eaob_profile, meas_noise=self.meas_noise,
                 noise_seed=self.noise_seed, tau_dist=float(P.EAOB_TAU_DIST),
                 mpc_state_source=self.mpc_state_source,
                 sig_pos_xy=float(P.EAOB_SIG_POS[0]))   # 2026-07-24: 0.05 -> 0.005 m
        if self.eaob is not None:
            m["gate"] = self.eaob.gate               # None = gate off
            m["n_upd"] = int(self.eaob.n_upd)
            m["n_gated"] = int(self.eaob.n_gated)
        return m

    def status(self):
        tag = "DOB-MPC" if self.mode == "dobmpc" else "MPC"
        wn = np.linalg.norm(self.w_hat[:3])
        gated = self.eaob.n_gated if self.eaob is not None else 0
        return f"{tag}  |w_hat|={wn:.1f}N  solve_fail={self.n_fail}  gated={gated}"
