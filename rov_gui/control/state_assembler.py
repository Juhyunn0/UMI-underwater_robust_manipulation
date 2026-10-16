#!/usr/bin/env python3
"""state_assembler.py — NavFix + VehicleImu -> the (eta, nu, nudot) the EAOB eats.

The EAOB cannot run pose-only: its measurement vector needs eta AND nu AND
nudot (dobmpc/eaob.py:108-129 — nudot is the entire mechanism that makes the
disturbance observable). This module assembles all three at the control tick,
in the controller's native NED/FRD, from the two sensors the vehicle has:

    eta[0:2] x, y        tag PnP (NavFix), ZOH between camera frames
    eta[2]   z           pressure depth + a session offset (default), or tag z
    eta[3:5] roll, pitch ArduSub ATTITUDE (gravity-referenced — always better
                         than a grazing-angle tag PnP for these two)
    eta[5]   yaw         tag PnP (the compass is a consistency check ONLY —
                         steel pool walls and thruster currents corrupt it)
    nu[0:3]  u, v, w     finite-difference of the tag position, low-passed
                         (vel_lp_alpha), rotated into the body frame — or,
                         for u and v only, a DVL's direct measurement when
                         the caller hands ``step`` a ``vel_xy``
                         (control/dvl.py decides per tick; UNVERIFIED on
                         hardware)
    nu[3:6]  p, q, r     ArduSub gyro (ATTITUDE rollspeed/pitchspeed/yawspeed)
    nudot    a           SCALED_IMU2 specific force + gravity - omega x v
                         ("imu" mode), or finite-difference of nu ("fd" mode
                         — the sim's definition, much noisier here because nu
                         itself is differentiated position)

Angle conventions match dobmpc.fossen.rot_ib = Rz(psi) Ry(theta) Rx(phi);
the composed R_ned_body uses the tag yaw with the autopilot's roll/pitch, so
the one rotation matrix used for velocity/gravity mapping is the same one
eta reports. This module stays dobmpc-free (plain numpy) so it imports —
and is testable — without acados/casadi installed.
"""

from __future__ import annotations

import math

import numpy as np

G_NED = np.array([0.0, 0.0, 9.80665])


def rot_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body(FRD)->NED rotation, Rz(psi) Ry(theta) Rx(phi) — matches
    dobmpc.fossen.rot_ib (kept local so this module needs no dobmpc)."""
    cf, sf = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(psi), math.sin(psi)
    return np.array([
        [cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
        [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
        [-st, ct * sf, ct * cf],
    ])


class StateAssembler:
    """Stateful: keeps the velocity filter, the pressure-z offset and the
    previous samples. One instance per engage session is the intended use;
    ``reset()`` returns it to cold."""

    def __init__(self, z_source: str = "pressure", vel_lp_alpha: float = 0.6,
                 nudot_source: str = "imu", tag_stale_s: float = 0.5,
                 imu_stale_s: float = 0.3, propagate: bool = True):
        assert z_source in ("pressure", "tag"), z_source
        assert nudot_source in ("imu", "fd"), nudot_source
        self.z_source = z_source
        self.alpha = float(vel_lp_alpha)
        self.nudot_source = nudot_source
        self.tag_stale_s = float(tag_stale_s)
        self.imu_stale_s = float(imu_stale_s)
        # Bridge BETWEEN camera frames: yaw by the gyro, x/y by the filtered
        # velocity. Without it the position the loop (and the plot) sees is a
        # per-frame ZOH — visibly "slow" at 10-20 Hz fix rates. Velocity-hold
        # over <=stale_s gaps errs by |v_err|*gap (~5 mm at 0.05 m/s, 100 ms);
        # accelerometer double-integration is deliberately NOT used (the
        # BNO086 carries a measured 1.80 m/s^2 bias — drift would
        # exceed the gap error).
        #
        # Still true HERE, and NOT contradicted by imu_dr.py, which does
        # double-integrate: this is a 100 ms hole in a stream that keeps being
        # corrected, that is an unaided run of tens of seconds whose
        # divergence is the quantity being measured. Different question, and a
        # much larger expected answer.
        self.propagate = bool(propagate)
        self.reset()

    def reset(self) -> None:
        self._prev_fix_t = None
        self._prev_p = None
        self._v_ned = np.zeros(3)           # low-passed world velocity
        self._prev_gyro = None
        self._prev_nu = None
        self._z_off = None                  # z_tag_world - depth_pressure
        self.rp_residual_deg = None         # tag-vs-ATTITUDE roll/pitch check
        self.rp_residual_rp_deg = None      # ...signed, (roll, pitch)

    # ------------------------------------------------------------------ z
    @property
    def z_offset(self) -> float | None:
        """``z_world_tag - depth_pressure``, once anchored. Public because the
        dead reckoner (imu_dr.py) needs the SAME offset to put the barometer
        in the same world — two independent anchors would put the two
        estimates at two different depths and the comparison would be of the
        anchoring, not of the IMU."""
        return self._z_off

    def calibrate_z_offset(self, fix, imu) -> bool:
        """Anchor pressure depth to the tag world once (call at engage, at
        rest). z_world = depth_pressure + z_off thereafter."""
        if fix is None or not fix.ok or imu is None or imu.depth_m is None:
            return False
        self._z_off = float(fix.p_ned[2]) - float(imu.depth_m)
        return True

    # --------------------------------------------------------------- step
    def step(self, fix, imu, t_now: float, dt: float, vel_xy=None,
             disp_xy=None):
        """-> (meas dict for EAOB.update | None, health dict).

        ``meas`` is None whenever either sensor is missing or stale — the
        caller decides what that means (refuse engage / disengage). ``fix``
        may be the SAME object across several ticks (camera slower than the
        control loop); the velocity filter only updates on a NEW t_capture.

        ``vel_xy`` (2026-10-01, control/dvl.py) is a MEASURED body-origin
        (surge, sway) velocity in FRD — a DVL's, already rotated by its
        mount and lever-arm corrected — or None. Given, it REPLACES the tag
        finite difference as nu[0:2], in the fix-age bridging and in the
        nudot transport term. The HEAVE rate nu[2] stays the finite
        difference either way: nothing verifies a DVL's vertical axis yet
        (dvl.py). The finite-difference filter keeps running underneath
        (``health["nu_fd"]``) so it is warm the tick the DVL drops out and
        the two can be compared on every row.

        ``disp_xy`` (only with ``vel_xy``) is the body-frame INTEGRAL of that
        velocity from the fix's capture time to now. With it the bridge is
        p_tag + R @ integral — a real integral — instead of "the newest
        velocity x the age", which is not one once the velocity keeps
        updating while the fix grows old (review 2026-10-02: at a kick the
        estimate ran ahead and then back). None = velocity x age.

        ``vel_xy=None`` is the pre-DVL path with the same operations in the
        same order, so ``meas`` is bit-identical to a build without these
        arguments (pinned by test_dvl). Position source, yaw, z, roll/pitch
        and the rates are NOT touched either way: a DVL is a velocity sensor.
        """
        h = {"tag_age": None, "imu_age": None, "ok": False, "why": ""}
        # The attitude's TRUE age (2026-10-15, see att_age_v below), set
        # BEFORE every early return: a tick with no tag fix still has an
        # attitude — the station bridge flies on it, the pwm LOWs' AGE
        # interlock judges it and the engage refusal pools it. It decides
        # nothing here: `imu_age` and the imu_stale_s rule are unchanged.
        h["att_age_v"] = att_age_v(imu, t_now)
        if fix is None or not fix.ok:
            h["why"] = "no tag fix"
            return None, h
        if imu is None or imu.roll is None or imu.p is None:
            h["why"] = "no vehicle imu"
            return None, h
        h["tag_age"] = max(0.0, t_now - float(fix.t_capture))
        imu_t = imu.t_att if imu.t_att is not None else imu.stamp
        h["imu_age"] = max(0.0, t_now - float(imu_t))
        if h["tag_age"] > self.tag_stale_s:
            h["why"] = f"tag fix stale ({h['tag_age']:.2f}s)"
            return None, h
        if h["imu_age"] > self.imu_stale_s:
            h["why"] = f"imu stale ({h['imu_age']:.2f}s)"
            return None, h

        p_tag = np.asarray(fix.p_ned, float)
        R_tag = np.asarray(fix.R_ned_body, float).reshape(3, 3)
        psi = math.atan2(R_tag[1, 0], R_tag[0, 0])
        # gyro bridging of yaw across the fix age (see __init__.propagate)
        age = 0.0
        if self.propagate:
            age = min(max(0.0, t_now - float(fix.t_capture)), self.tag_stale_s)
            if imu.r is not None:
                psi += float(imu.r) * age
        phi, theta = float(imu.roll), float(imu.pitch)
        R = rot_zyx(phi, theta, psi)        # THE rotation for everything below

        # Stationary consistency diagnostic: the tag pose implies a roll/pitch
        # too; a mounting/preset error shows up here long before it shows up
        # as a bad trajectory. Reported, never fused.
        th_tag = math.asin(max(-1.0, min(1.0, -float(R_tag[2, 0]))))
        ph_tag = math.atan2(float(R_tag[2, 1]), float(R_tag[2, 2]))
        d_phi, d_th = _wrap(ph_tag - phi), _wrap(th_tag - theta)
        self.rp_residual_deg = math.degrees(max(abs(d_phi), abs(d_th)))
        # SIGNED and SEPARATE, because the magnitude alone cannot be acted on:
        # `cam_tilt_deg` corrects a PITCH, so an operator reading 43 deg needs
        # to know whether that is 43 of pitch (enter it) or a mix with roll
        # (the mount is skewed as well as tilted, and one number will not fix
        # it). The max() above is kept because the CSV and older readers use it.
        self.rp_residual_rp_deg = (math.degrees(d_phi), math.degrees(d_th))

        # ---- position (z per config)
        z = float(p_tag[2])
        z_src = "tag"
        if self.z_source == "pressure":
            # The barometer has its OWN stream: ATTITUDE staying fresh says
            # nothing about SCALED_PRESSURE2, and a frozen depth under a live
            # loop is exactly the silent failure the Freshness rule bans.
            # 1.5 s is generous for a >=10 Hz baro and still catches a death.
            baro_t = imu.t_baro
            if (imu.depth_m is None or baro_t is None
                    or t_now - float(baro_t) > 1.5):
                h["why"] = "pressure depth stale"
                return None, h
            if self._z_off is None:
                self._z_off = z - float(imu.depth_m)   # self-anchor on first use
            z = float(imu.depth_m) + self._z_off
            z_src = "pressure"
        p = np.array([p_tag[0], p_tag[1], z])

        # ---- world velocity from tag deltas (only on a NEW camera frame)
        #
        # The low-pass is RECURSIVE, so alpha is not a weighting — it decides
        # how long the past lingers. (1 - alpha) survives every step, which
        # puts the output's centre of mass (1 - alpha)/alpha frames back:
        # 124 ms at the old 0.35, 44 ms at 0.6 (fix interval 0.0666 s
        # measured). That lag is multiplied by kd = 59.7 N.s/m in the PID and
        # is the estimator's dominant error at this vehicle's speeds — the
        # measured position noise is 0.8 mm, so smoothing was buying almost
        # nothing for it. Provenance for both numbers: config/hw_mpc.yaml.
        tc = float(fix.t_capture)
        if self._prev_fix_t is not None and tc > self._prev_fix_t + 1e-6:
            dtc = tc - self._prev_fix_t
            v_new = (p_tag - self._prev_p) / dtc
            a = self.alpha
            self._v_ned = (1.0 - a) * self._v_ned + a * v_new
        if self._prev_fix_t is None or tc > self._prev_fix_t + 1e-6:
            self._prev_fix_t, self._prev_p = tc, p_tag.copy()

        # velocity bridging of x/y (and tag-z) across the fix age
        nu_lin = R.T @ self._v_ned
        h["nu_fd"] = nu_lin              # the tag finite difference, always
        h["vel_src"] = "tag_fd"
        if vel_xy is None:
            if age > 0.0:
                p[0] += self._v_ned[0] * age
                p[1] += self._v_ned[1] * age
                if z_src == "tag":
                    p[2] += self._v_ned[2] * age
        else:
            vel_xy = np.asarray(vel_xy, float).reshape(2)
            # (u, v) measured, w still the finite difference's
            nu_lin = np.array([vel_xy[0], vel_xy[1], nu_lin[2]])
            h["vel_src"] = "dvl"
            if age > 0.0:
                if disp_xy is not None:
                    d = np.asarray(disp_xy, float).reshape(2)
                    step = R @ np.array([d[0], d[1], nu_lin[2] * age])
                else:
                    step = (R @ nu_lin) * age
                p[0] += step[0]
                p[1] += step[1]
                if z_src == "tag":
                    p[2] += step[2]
        gyro = np.array([float(imu.p), float(imu.q), float(imu.r)])
        nu = np.concatenate([nu_lin, gyro])

        # ---- nudot
        if self._prev_gyro is None:
            ang_acc = np.zeros(3)
        else:
            ang_acc = (gyro - self._prev_gyro) / dt
        self._prev_gyro = gyro.copy()
        if self.nudot_source == "imu" and imu.ax is not None:
            f = np.array([float(imu.ax), float(imu.ay), float(imu.az)])
            # accelerometer measures specific force f = a_world_in_body - R^T g
            # => body-frame translational acceleration a = f + R^T g; the
            # body-rate term converts it to d(nu)/dt (transport theorem).
            a_lin = f + R.T @ G_NED - np.cross(gyro, nu_lin)
        else:
            if self._prev_nu is None:
                a_lin = np.zeros(3)
            else:
                a_lin = (nu_lin - self._prev_nu[:3]) / dt
        self._prev_nu = nu.copy()
        nudot = np.concatenate([a_lin, ang_acc])

        eta = np.array([p[0], p[1], p[2], phi, theta, psi])
        h["ok"] = True
        h["z_src"] = z_src
        h["rp_residual_deg"] = self.rp_residual_deg
        h["rp_residual_rp_deg"] = self.rp_residual_rp_deg
        return {"eta": eta, "nu": nu, "nudot": nudot}, h


def att_age_v(imu, t_now: float) -> float | None:
    """How old the ATTITUDE in ``imu`` (roll, pitch, p, q, r) is at ``t_now``,
    on the VEHICLE's clock: ``t_now - imu.t_att_v``, where ``t_att_v`` is that
    sample's time_boot_ms put on the host clock by the MAVLink reader's
    lower-envelope map (c3_camera/mavlink_log.VehicleClockMap) — 2026-10-15,
    RL_controller/logs/eval/pool_trips_1013_rootcause.

    ``imu_age`` (``t_now - t_att``) starts the clock at the station's PARSE
    time, so a queue on the vehicle side of the link is invisible to it: on
    the 2026-10-13 engagements #1, #3, #4, #5 the CSV imu_age_s read p50
    0.054-0.058 s while this age read p50 0.35-0.51 s [측정: pool_trips_1013_
    rootcause/synth/out/synth_checks.json C1_age].

    None = unknown (no snapshot, no map, a non-finite stamp) — never 0: the
    pwm LOWs count an unknown age as over (rl_pwm.PwmInterlocks). Floored at 0
    like ``imu_age`` (a synthetic stamp may sit ahead of ``t_now``)."""
    t_v = getattr(imu, "t_att_v", None) if imu is not None else None
    if t_v is None:
        return None
    try:
        a = float(t_now) - float(t_v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a):
        return None
    return max(0.0, a)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
