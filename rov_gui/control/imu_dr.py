#!/usr/bin/env python3
"""imu_dr.py — strapdown dead reckoning from the C3's IMU, and nothing else.

The question this module exists to answer is "how far can the IMU carry the
vehicle on its own", so it is deliberately NOT a fusion filter: once anchored
it never looks at a tag again. Its output is compared against the AprilTag
pose, and the rate at which the two separate IS the measurement.

    tag anchor (once, at the end of settle)
        |
        v
    accel + gyro @ ~200 Hz  ->  attitude  ->  a_ned = R f + g  ->  v  ->  p

Why this does not contradict state_assembler
--------------------------------------------
``state_assembler.py`` says, correctly, that accelerometer double-integration
is deliberately not used to BRIDGE between camera frames: over a 100 ms gap
the velocity-hold error is ~5 mm and the double-integration error is worse.
That statement is about a 100 ms hole in a stream that keeps being corrected.
This module integrates for tens of seconds with no correction at all, which is
a different question with a different — and much larger — expected answer.
Both can be true and both are.

Frames and conventions (the same ones, deliberately)
----------------------------------------------------
Everything is NED/FRD and every rotation comes from
``state_assembler.rot_zyx`` so there is exactly one angle convention in the
package. ``G_NED = [0, 0, +9.80665]``. The accelerometer reports SPECIFIC
FORCE: a level, still vehicle reads ``f = (0, 0, -9.81)`` in FRD, so

    a_ned = R_ned_body @ f_frd + G_NED

which is zero at rest, and which is the same identity ``state_assembler``
inverts at its line 200.

What is observable and what is not
----------------------------------
Three error sources, and they are not equally fixable:

* **Accel bias** ``b``            -> position error ``0.5 * b * t^2``
* **Attitude error** ``theta``    -> ``0.5 * g * sin(theta) * t^2``
* **Gyro bias** ``beta``, pure    -> ``(1/6) * g * beta * t^3``
* **Accel scale** ``s`` over ``dv`` -> ``s * dv * t`` (linear, never recovers)

The BNO086 on this camera carries a **1.80 m/s^2 (0.184 g) accelerometer
bias, almost entirely on IMU x**, with the scale within 0.9% of unity
[측정: 6-attitude tumble, sessions/low_level_controller_data/20260817/
0817_101511 + _100139, 21985 still samples; config/c3_imu_calib.json
sha1 7081ff43]. Uncorrected that is 0.5 * 1.80 * t^2 = **90 m at 10 s** — the
estimator is not approximately wrong without a calibration, it is useless.

(This repo long recorded the defect as a "+20% SCALE error", from a single
upright reading of |a| = 11.8. It was a misreading: in that pose gravity lies
nearly along IMU x, so the bias adds straight onto it. The correction matters
because the two behave differently — a constant body-frame bias is removed
COMPLETELY by the static correction below, at any attitude, while a scale
error would keep leaking as the attitude changed.)

Scale and bias are NOT separable from a single orientation, which is why there
are two mechanisms: a bench ellipsoid fit that separates them properly
(``rov_gui/tools/calib_c3_imu.py``), and a per-run static correction that
lumps them together at the attitude the vehicle actually settled at. The
run-time one always applies; the bench one is what keeps its residual small
once the vehicle leaves that attitude.

The IMU-to-body rotation is estimated directly (Kabsch against the autopilot's
body rates, in the calibration tool) rather than composed out of an
IMU-to-camera extrinsic and a camera mount angle. The device has no
IMU-to-camera extrinsic at all — ``getImuToCameraExtrinsics()`` raises — and
going straight to the body frame means the camera's tilt is absorbed
automatically and this module never needs to know where the camera points.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from .state_assembler import G_NED, rot_zyx

G = float(G_NED[2])

# One IMU sample per row. Kept as a plain float64 array rather than a list of
# objects because it crosses a thread boundary at ~200 Hz and gets sliced far
# more often than it gets inspected.
SAMPLE_COLS = ("t", "ax", "ay", "az", "gx", "gy", "gz")
N_COLS = len(SAMPLE_COLS)

ATTITUDE_MODES = ("gyro", "ahrs", "vehicle")


def euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) from a body(FRD)->NED rotation — inverse of
    ``rot_zyx``, and the same extraction ``state_assembler`` uses for the tag
    pose at its lines 135-136."""
    theta = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    phi = math.atan2(float(R[2, 1]), float(R[2, 2]))
    psi = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return phi, theta, psi


def _skew(v) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rodrigues(w, dt: float) -> np.ndarray:
    """Exact exp(skew(w)*dt). Rodrigues rather than the first-order update
    because at 200 Hz the truncation is small but SYSTEMATIC, and a systematic
    attitude error is exactly what this module is trying to measure in the
    sensor rather than manufacture in itself."""
    v = np.asarray(w, float) * float(dt)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    K = _skew(v / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Pull R back onto SO(3). Repeated multiplication drifts off the manifold
    slowly; over a 200 Hz, minutes-long integration "slowly" is still enough to
    matter, and an R with det != 1 quietly rescales gravity."""
    U, _s, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


# =============================================================================
# calibration
# =============================================================================
class ImuCalibration:
    """Scale, bias and mounting rotation for one IMU, from a JSON on disk.

    ``identity()`` is a usable object, not a null one: it means "no bench
    calibration was applied", which the run meta records as
    ``calibration_sha1: null``. That distinction has to survive into the
    recording, because a drift curve measured without a calibration is
    measuring something else.
    """

    def __init__(self, accel_scale=None, accel_bias=None, gyro_bias=None,
                 R_frd_imu=None, path: str = "", sha1: str = "",
                 note: str = "", raw: dict | None = None):
        self.accel_scale = (np.ones(3) if accel_scale is None
                            else np.asarray(accel_scale, float).reshape(3))
        self.accel_bias = (np.zeros(3) if accel_bias is None
                           else np.asarray(accel_bias, float).reshape(3))
        self.gyro_bias = (None if gyro_bias is None
                          else np.asarray(gyro_bias, float).reshape(3))
        self.R_frd_imu = (np.eye(3) if R_frd_imu is None
                          else orthonormalize(
                              np.asarray(R_frd_imu, float).reshape(3, 3)))
        self.path = str(path)
        self.sha1 = str(sha1)
        self.note = str(note)
        self.raw = dict(raw or {})

    @classmethod
    def identity(cls) -> "ImuCalibration":
        return cls(note="no calibration file — raw IMU, scale and mounting "
                        "rotation both unverified")

    @classmethod
    def from_json(cls, path) -> "ImuCalibration":
        p = Path(path)
        blob = p.read_bytes()
        d = json.loads(blob.decode("utf-8"))
        acc = d.get("accel") or {}
        gyr = d.get("gyro") or {}
        return cls(accel_scale=acc.get("scale"),
                   accel_bias=acc.get("bias"),
                   gyro_bias=gyr.get("bias"),
                   R_frd_imu=d.get("R_frd_imu"),
                   path=str(p),
                   sha1=hashlib.sha1(blob).hexdigest(),
                   note=d.get("note", ""),
                   raw=d)

    def correct(self, a_raw: np.ndarray, g_raw: np.ndarray):
        """(N,3),(N,3) in the IMU frame -> the same in body FRD, scaled and
        de-biased. The mounting rotation is applied LAST so scale and bias stay
        expressed in the axes the ellipsoid fit measured them on."""
        a = (np.asarray(a_raw, float) * self.accel_scale) - self.accel_bias
        return a @ self.R_frd_imu.T, np.asarray(g_raw, float) @ self.R_frd_imu.T

    def meta(self) -> dict:
        return {
            "path": self.path or None,
            "sha1": self.sha1 or None,
            "accel_scale": [float(v) for v in self.accel_scale],
            "accel_bias": [float(v) for v in self.accel_bias],
            "gyro_bias": (None if self.gyro_bias is None
                          else [float(v) for v in self.gyro_bias]),
            "R_frd_imu": [[float(v) for v in row] for row in self.R_frd_imu],
            "note": self.note or None,
        }


# =============================================================================
# the dead reckoner
# =============================================================================
class ImuDeadReckoner:
    """Stateful. One instance per engage session; ``reset()`` returns it cold.

    Lifecycle, in the order the worker drives it:

        integrate(batch)        continuously, from the moment samples arrive.
                                Before the anchor this only fills the static
                                window — nothing is propagated.
        calibrate_static(R)     once, at the end of settle: gyro bias and the
                                static accel correction come out of the window.
        anchor(eta, nu, t)      once, at the same moment: position, velocity
                                and attitude are seeded from the tag state.
        integrate(batch)        thereafter this is the whole estimator.
        state(t, meas_tag)      at the control rate.
    """

    def __init__(self, calib: ImuCalibration | None = None,
                 attitude: str = "ahrs", ahrs_tau_s: float = 3.0,
                 accel_trust_m_s2: float = 0.15, z_source: str = "pressure",
                 max_dt_s: float = 0.1, static_window_s: float = 8.0,
                 gyro_static_std_max: float = 0.02,
                 gyro_bias_sem_max: float = 0.005,
                 accel_static_sd_max: float = 0.15,
                 stale_s: float = 0.5):
        assert attitude in ATTITUDE_MODES, attitude
        assert z_source in ("pressure", "imu"), z_source
        self.calib = calib or ImuCalibration.identity()
        self.attitude = attitude
        self.ahrs_tau_s = float(ahrs_tau_s)
        self.accel_trust = float(accel_trust_m_s2)
        self.z_source = z_source
        self.max_dt_s = float(max_dt_s)
        self.static_window_s = float(static_window_s)
        self.gyro_static_std_max = float(gyro_static_std_max)
        self.gyro_bias_sem_max = float(gyro_bias_sem_max)
        self.accel_static_sd_max = float(accel_static_sd_max)
        self.stale_s = float(stale_s)
        self.reset()

    # ------------------------------------------------------------------ life
    def reset(self) -> None:
        self.anchored = False
        self.t_anchor = None
        self.R = np.eye(3)                  # body FRD -> NED, estimated
        self.p = np.zeros(3)                # NED, datum frame
        self.v = np.zeros(3)                # NED
        self.w = np.zeros(3)                # last corrected body rate
        self.f = np.zeros(3)                # last corrected specific force
        self.a_ned = np.zeros(3)
        self._t_last = None
        self._prev_w = None
        self._static: deque = deque()       # pre-anchor window, (N,7) chunks
        self._c_static = np.zeros(3)        # body-frame accel correction
        self._b_gyro = np.zeros(3)
        self.gyro_bias_source = "none"
        self.accel_ref_source = "none"
        self.static_quality: dict = {}
        self.static_ok = False
        self.gyro_ok = False
        self.static_note = ""
        self.n_samples = 0                  # integrated since anchor
        self.n_rejected = 0                 # dt out of range
        self.n_dropped = 0                  # reported lost by the transport
        self.n_ungated = 0                  # AHRS levelling skipped
        self._marks: deque = deque(maxlen=1024)   # arrival times, for hz
        self._t_newest = None
        self._z = 0.0                       # pressure depth (NED, +down)
        self._vz = 0.0                      # ...and its low-passed rate
        self._z_t = None
        self._z_have = False
        self.note = ""

    # --------------------------------------------------------------- ingest
    def integrate(self, arr, dropped: int = 0) -> int:
        """Push one batch, shape (N, 7) in SAMPLE_COLS order. Returns how many
        samples were actually propagated."""
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[1] != N_COLS or a.shape[0] == 0:
            return 0
        self.n_dropped += int(dropped)
        self._t_newest = float(a[-1, 0])
        self._marks.append((self._t_newest, a.shape[0]))
        if not self.anchored:
            self._static.append(a)
            self._prune_static()
            return 0
        return self._propagate(a)

    def _prune_static(self) -> None:
        if not self._static:
            return
        newest = float(self._static[-1][-1, 0])
        while len(self._static) > 1:
            if newest - float(self._static[0][-1, 0]) > self.static_window_s:
                self._static.popleft()
            else:
                break

    def static_window(self) -> np.ndarray:
        """Everything currently in the pre-anchor window, oldest first."""
        if not self._static:
            return np.zeros((0, N_COLS))
        return np.concatenate(list(self._static), axis=0)

    # ---------------------------------------------------------- calibration
    def calibrate_static(self, R_body_ned: np.ndarray,
                         gyro_ref: np.ndarray | None = None,
                         att_ref: np.ndarray | None = None) -> dict:
        """Consume the settle window. ``R_body_ned`` is the attitude the
        vehicle held at the anchor instant (from the tag/ATTITUDE state);
        ``att_ref`` is that attitude as an (N, 3) [t, roll, pitch] SERIES over
        the window, and supplying it is what makes ``c_static`` correct.

        Two things come out, and they fail differently:

        * ``c_static`` — the body-frame offset between the specific force the
          accelerometer reported and the one gravity says it should have
          reported at that attitude. This absorbs bias AND the +20% scale at
          that one attitude, and it is applied unconditionally because without
          it nothing else in this module is worth running.
        * ``b_gyro`` — the gyro bias, IF the calibration file did not already
          supply one. A DP-holding vehicle has live thrusters and is not
          actually still, so a real yaw drift would be absorbed as "bias" here.
          Pass ``gyro_ref`` (the autopilot's body rates over the same window)
          to difference that out, or put a deck-measured bias in the JSON and
          let this be the check it is better suited to being.
        """
        win = self.static_window()
        out = {"n": int(win.shape[0]), "ok": False, "why": "",
               "accel_ok": False, "gyro_ok": False}
        if win.shape[0] < 10:
            out["why"] = f"only {win.shape[0]} samples in the settle window"
            self.static_note = out["why"]
            return out

        f_b, w_b = self.calib.correct(win[:, 1:4], win[:, 4:7])
        tw = win[:, 0]
        f_sd = float(np.linalg.norm(f_b, axis=1).std())
        w_std = w_b.std(axis=0)
        out["gyro_std"] = [float(v) for v in w_std]
        out["accel_mag_std"] = f_sd
        out["accel_mag_mean"] = float(np.linalg.norm(f_b, axis=1).mean())
        why = []

        # ---- gyro bias. Gated on the PRECISION OF ITS MEAN, not on how much
        # the gyro wobbled. Those are different questions and conflating them
        # threw away every real settle window: measured 2026-08-18, a vehicle
        # holding station at 0.1 cm/s showed 1.40 deg/s of gyro STD (thruster
        # vibration) while its mean was determined to 0.026 deg/s. The old
        # std gate rejected that, no bias was subtracted, and the resulting
        # 0.128 deg/s bias became a 1.28 deg standing tilt under the AHRS —
        # 0.22 m/s^2 of leaked gravity and 37.6 m of drift in 19 s
        # [측정: sessions/low_level_controller_data/20260818/0818_151119].
        #
        # And when gyro_ref is supplied the rotation question does not arise
        # at all: differencing against the autopilot removes real motion,
        # which is the whole reason it is passed in.
        # gyro_ref may be a single (3,) mean OR an (N, 4) [t, p, q, r] series.
        # The SERIES is what actually works, and the difference is not
        # cosmetic: subtracting a constant shifts the mean but leaves the
        # vehicle's real time-varying rotation in the residual, so the chunk
        # scatter — the very thing the precision gate reads — still measures
        # the vehicle instead of the sensor. Measured on one settle window:
        # 0.506 deg/s with a constant reference, 0.026 deg/s with the series.
        ref = None if gyro_ref is None else np.asarray(gyro_ref, float)
        if ref is not None and ref.ndim == 2 and ref.shape[1] == 4:
            ref = np.stack([np.interp(tw, ref[:, 0], ref[:, k + 1])
                            for k in range(3)], axis=1)
            out["gyro_ref"] = "autopilot rate series"
        elif ref is not None:
            ref = ref.reshape(3)
            out["gyro_ref"] = "autopilot rate, single sample"
        d = w_b if ref is None else (w_b - ref)
        n_ch = max(2, int((tw[-1] - tw[0]) // 1.0))
        edges = np.linspace(tw[0], tw[-1], n_ch + 1)
        ch = [d[(tw >= a) & (tw < b)].mean(axis=0)
              for a, b in zip(edges[:-1], edges[1:])]
        ch = np.array([c for c in ch if np.all(np.isfinite(c))])
        sem = (float(np.linalg.norm(ch.std(axis=0)) / math.sqrt(len(ch)))
               if len(ch) >= 2 else float("inf"))
        out["gyro_bias_sem"] = sem
        if sem > self.gyro_bias_sem_max:
            why.append(f"gyro bias undetermined (sem {sem:.5f} > "
                       f"{self.gyro_bias_sem_max:.5f} rad/s)")
        elif ref is None and float(w_std.max()) > self.gyro_static_std_max:
            # No reference to difference against, so stillness IS required.
            why.append(f"gyro not still and no autopilot reference "
                       f"(std {float(w_std.max()):.4f} > "
                       f"{self.gyro_static_std_max:.4f} rad/s)")
        else:
            if self.calib.gyro_bias is not None:
                # ROTATED. calib.correct() applies the mounting rotation LAST
                # so that scale and bias stay in the axes the bench fit
                # measured them on (ImuCalibration.correct), which means the
                # rates _propagate subtracts this from are already FRD while
                # the JSON's bias is still in IMU axes. R_frd_imu is ~48 deg
                # here, so subtracting it unrotated would put a deck-measured
                # bias on the wrong axes entirely — latent until now only
                # because no calibration file has ever carried a gyro block.
                self._b_gyro = self.calib.R_frd_imu @ self.calib.gyro_bias
                self.gyro_bias_source = "calibration (deck), rotated to FRD"
                out["gyro_bias_check"] = [float(v) for v in d.mean(axis=0)]
            elif ref is not None:
                self._b_gyro = d.mean(axis=0)
                self.gyro_bias_source = "settle window minus autopilot rates"
            else:
                self._b_gyro = d.mean(axis=0)
                self.gyro_bias_source = "settle window mean (assumes still)"
            out["gyro_ok"] = True
            out["b_gyro"] = [float(v) for v in self._b_gyro]
            out["gyro_bias_source"] = self.gyro_bias_source

        # ---- accel static correction. THIS one needs the vehicle to be
        # translationally still, because a hand or a thruster is
        # indistinguishable from gravity. Steadiness of |f| is the test —
        # measured 0.032 m/s^2 on a real DP settle, so 0.15 is generous.
        if f_sd > self.accel_static_sd_max:
            why.append(f"accel not steady (|a| std {f_sd:.3f} > "
                       f"{self.accel_static_sd_max:.3f} m/s^2)")
        else:
            # The offset is a PER-SAMPLE quantity: what this sample read minus
            # what gravity says it should have read AT THAT SAMPLE'S ATTITUDE.
            # Averaging f over the window and differencing one attitude is not
            # the same thing, and the gap is not small. A station-keeping
            # vehicle is attitude-live: measured over one 8 s settle, pitch
            # swung -0.34 -> +2.15 deg (std 0.75) while roll wandered 0.43 deg,
            # so the window's mean f belonged to the window's MEAN attitude
            # while the anchor R was the END attitude. The 0.5 deg gap became
            # 0.12 m/s^2 of permanent bias, which under the AHRS is a fixed
            # 0.36 m/s velocity error and a LINEAR 4.3 m of drift by 17 s.
            # Per-sample: 0.74 m over the same 17 s, from the same log
            # [측정: sessions/low_level_controller_data/20260818/0818_160139,
            #  mpc_161904 — offline A/B on the flown samples].
            ref = None if att_ref is None else np.asarray(att_ref, float)
            if ref is not None and ref.ndim == 2 and ref.shape[1] >= 3:
                phi = np.interp(tw, ref[:, 0], ref[:, 1])
                the = np.interp(tw, ref[:, 0], ref[:, 2])
                # -R(phi,theta)^T g, written out so it costs one pass and no
                # per-sample matrix build. Yaw cancels: R_z^T e_z = e_z.
                cf, sf = np.cos(phi), np.sin(phi)
                ct, st = np.cos(the), np.sin(the)
                g = float(G_NED[2])
                expect = np.column_stack([g * st, -g * ct * sf, -g * ct * cf])
                out["accel_ref"] = "autopilot attitude series"
            else:
                R = orthonormalize(np.asarray(R_body_ned, float).reshape(3, 3))
                expect = np.broadcast_to(-R.T @ G_NED, f_b.shape)
                out["accel_ref"] = "anchor attitude only (single sample)"
            self._c_static = (f_b - expect).mean(axis=0)
            self.accel_ref_source = out["accel_ref"]
            out["accel_ok"] = True
            out["c_static"] = [float(v) for v in self._c_static]

        # Independent on purpose: losing one is no reason to lose the other,
        # and the old all-or-nothing refusal is exactly how a good gyro bias
        # got discarded along with a marginal accel window.
        self.static_ok = bool(out["accel_ok"])
        self.gyro_ok = bool(out["gyro_ok"])
        out["ok"] = bool(out["accel_ok"] and out["gyro_ok"])
        out["why"] = "; ".join(why)
        self.static_note = out["why"]
        # Carried into the run meta. Without them a later reader can see THAT
        # the window was accepted but not how well it determined anything, and
        # "accepted" is the weaker statement: measured 2026-08-18, two settle
        # windows 18 min apart passed the SEM gate with 27x and 50x margin and
        # still disagreed on the yaw-gyro bias by 68%.
        self.static_quality = {
            "n": out["n"], "gyro_bias_sem": out.get("gyro_bias_sem"),
            "gyro_std": out.get("gyro_std"),
            "accel_mag_mean": out.get("accel_mag_mean"),
            "accel_mag_std": out.get("accel_mag_std"),
            "gyro_ref": out.get("gyro_ref"), "accel_ref": out.get("accel_ref")}
        return out

    # -------------------------------------------------------------- anchor
    def anchor(self, eta, nu_world_ned=None, t: float | None = None,
               zero_velocity: bool = True) -> None:
        """Seed the state from the tag solution. ``eta`` is the 6-vector the
        state assembler produces (NED position + roll/pitch/yaw), already
        datumized. ``nu_world_ned`` is the WORLD-frame velocity, used only when
        ``zero_velocity`` is False.

        Velocity is the anchor term worth being fussy about: the tag velocity
        is a low-passed finite difference carrying ~44 ms of lag, and 0.01 m/s
        of error at the anchor is 0.1 m by 10 s all on its own — the same order
        as everything else in the budget. At the end of a settle the vehicle is
        supposed to be stopped, so zero is both truer and quieter.
        """
        eta = np.asarray(eta, float).reshape(6)
        self.p = eta[:3].copy()
        self.R = rot_zyx(float(eta[3]), float(eta[4]), float(eta[5]))
        self.v = (np.zeros(3) if zero_velocity or nu_world_ned is None
                  else np.asarray(nu_world_ned, float).reshape(3).copy())
        self.a_ned = np.zeros(3)
        self._prev_w = None
        self.t_anchor = float(t) if t is not None else self._t_newest
        self._t_last = self._t_newest
        self.n_samples = 0
        self.n_rejected = 0
        self.n_ungated = 0
        self.anchored = True
        self._static.clear()

    # --------------------------------------------------------- mechanization
    def _propagate(self, arr: np.ndarray) -> int:
        f_all, w_all = self.calib.correct(arr[:, 1:4], arr[:, 4:7])
        f_all = f_all - self._c_static
        w_all = w_all - self._b_gyro
        n = 0
        for i in range(arr.shape[0]):
            t = float(arr[i, 0])
            if self._t_last is None:
                self._t_last = t
                continue
            dt = t - self._t_last
            if dt <= 0.0 or dt > self.max_dt_s:
                # Out-of-order or a hole big enough that integrating across it
                # is a guess. Skipping loses the motion; integrating invents
                # it. Losing it is the honest failure, and n_rejected says so.
                self._t_last = t
                self.n_rejected += 1
                continue
            f = f_all[i]
            w = w_all[i]
            # The levelling term is a FILTER correction, not a rate the
            # vehicle actually turned at. It steers the attitude integration
            # and nothing else — self.w, which becomes nu[3:6] for the
            # controller and the EAOB, stays the de-biased gyro.
            w_prop = self._level(w, f) if self.attitude == "ahrs" else w
            self.R = self.R @ rodrigues(w_prop, dt)
            if (n & 0x3F) == 0:
                self.R = orthonormalize(self.R)

            a = self.R @ f + G_NED
            # Trapezoid on velocity, and the matching 0.5*a*dt^2 on position:
            # at 0.05 m/s a rectangle rule would be invisible, but this module
            # is judged on t^2 growth and the integrator must not contribute
            # any of its own.
            self.p = self.p + self.v * dt + 0.5 * a * dt * dt
            self.v = self.v + a * dt
            self.a_ned = a
            self.f = f
            self.w = w
            self._t_last = t
            n += 1
        self.R = orthonormalize(self.R)
        self.n_samples += n
        return n

    def _level(self, w: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Mahony-style gravity levelling of roll and pitch. Yaw is untouched.

        ``d_b`` is where the estimate says "down" is, in body axes; ``m_b`` is
        where the accelerometer says it is. Their cross product is the small
        rotation between them, and it is perpendicular to ``d_b`` BY
        CONSTRUCTION — that is why this cannot touch yaw, and why no explicit
        yaw projection is needed. There is no magnetometer in the loop and
        there could not be one: the pool is steel and the thrusters are
        centimetres away.

        The gain is ``1/tau``: a tilt error decays as ``exp(-t/tau)``, and a
        gyro bias ``beta`` settles at a standing tilt of ``beta*tau`` [유도].
        That is the whole trade, and it is sharper than it looks:

        * levelling turns the gyro's ``(1/6) g beta t^3`` into a bounded
          ``0.5 g beta tau t^2``, so it only WINS for ``t >> 3*tau`` [유도] —
          at tau = 10 s the two are equal at 30 s;
        * but a small tau also rejects real horizontal acceleration below
          ``1/tau``, and this vehicle's whole signal is 1 s ramps at
          0.05 m/s^2. Eat those and the dead reckoner never learns it is
          moving, which is a worse error than the tilt it bought.

        10 s is the middle of that, not an optimum. It is recorded in the run
        meta because it changes what the drift curve means.
        """
        mag = float(np.linalg.norm(f))
        if mag < 1e-6 or abs(mag - G) > self.accel_trust:
            # Accelerating hard enough that "down" is not down. At this
            # vehicle's 0.05 m/s and 0.05 m/s^2 profile this should almost
            # never fire, so if n_ungated is large the run has a thruster or a
            # mounting problem worth looking at before the drift curve.
            self.n_ungated += 1
            return w
        m_b = -f / mag
        d_b = self.R.T @ (G_NED / G)
        return w + (np.cross(m_b, d_b) / self.ahrs_tau_s)

    # ---------------------------------------------------------------- output
    def hz(self, t_now: float, window_s: float = 2.0) -> float:
        cut = t_now - window_s
        n = sum(k for (ts, k) in self._marks if ts >= cut)
        return float(n) / window_s if n > 1 else 0.0

    def note_depth(self, z_ned: float | None, t_baro: float | None) -> None:
        """Take a pressure depth sample, and difference it for a heave rate.

        Depth is the one channel this experiment deliberately does NOT dead
        reckon (operator's decision, 2026-08-16): a diverging vertical channel
        would drive heave commands into the mat or the surface, and the
        question being asked is about horizontal drift.

        It comes from the BAROMETER and not from the tag solution, which
        matters more than it looks: the barometer is a different sensor on a
        different link, so a tag dropout leaves it running — and the ticks
        where the tag is gone are precisely the ones this experiment cares
        most about. Reading z out of the tag-assembled state instead made the
        estimator report "not ok" exactly when it was the only thing left.

        ``z_ned`` must already be in the frame this estimator was anchored in
        (the caller applies the session z-offset AND the mission datum) — a
        raw tag-world depth here would sit the vertical channel a datum apart
        from the horizontal one.
        """
        if z_ned is None:
            return
        t, z = t_baro, float(z_ned)
        if t is not None and self._z_t is not None and t > self._z_t + 1e-9:
            vz = (z - self._z) / (t - self._z_t)
            a = 0.35
            self._vz = (1.0 - a) * self._vz + a * vz
        if t is None or self._z_t is None or t > self._z_t + 1e-9:
            self._z_t = (float(t) if t is not None else None)
            self._z = z
        self._z_have = True

    def state(self, t_now: float, meas_tag: dict | None = None,
              imu_vehicle=None) -> dict:
        """The dead-reckoned state, in the same shape the controller eats.

        With ``z_source: pressure`` (the default and the operator's choice)
        depth and heave rate come from :meth:`note_depth`; ``meas_tag`` is
        only a last-resort fallback for a bench run with no barometer. The
        IMU's own integrated depth is still reported as ``pz_imu`` so the
        "could it have held depth?" comparison is free afterwards.
        """
        h = {"ok": False, "why": "", "meas": None,
             "elapsed": None, "hz": self.hz(t_now), "n": self.n_samples,
             "rejected": self.n_rejected, "ungated": self.n_ungated,
             "dropped": self.n_dropped, "pz_imu": float(self.p[2])}
        if not self.anchored:
            h["why"] = "not anchored"
            return h
        h["elapsed"] = max(0.0, t_now - float(self.t_anchor))
        if self._t_newest is None:
            h["why"] = "no imu samples"
            return h
        age = t_now - float(self._t_newest)
        h["sample_age"] = age
        if age > self.stale_s:
            # A dead reckoner starved of samples looks PERFECT — a frozen
            # point that tracks nothing and drifts not at all. It is the worst
            # failure this experiment can have, so it is a hard not-ok rather
            # than a note.
            h["why"] = f"imu samples stale ({age:.2f}s)"
            return h

        phi, theta, psi = euler_zyx(self.R)
        p = self.p.copy()
        v = self.v.copy()
        if self.attitude == "vehicle" and imu_vehicle is not None \
                and imu_vehicle.roll is not None:
            # Roll/pitch from the autopilot's own solution, yaw still ours:
            # ArduSub's compass is corrupted by the steel walls and the
            # thrusters (state_assembler.py:13-14), so taking its yaw would
            # ship a known-bad channel inside a mode meant to isolate errors.
            phi, theta = float(imu_vehicle.roll), float(imu_vehicle.pitch)
        if self.z_source == "pressure":
            if self._z_have:
                p[2], v[2] = self._z, self._vz
            elif meas_tag is not None:
                p[2] = float(meas_tag["eta"][2])
                R_tag = rot_zyx(*[float(x) for x in meas_tag["eta"][3:6]])
                v[2] = float((R_tag @ np.asarray(meas_tag["nu"], float)[:3])[2])
            else:
                h["why"] = "z_source pressure but no depth yet"
                return h

        R = rot_zyx(phi, theta, psi)
        nu_lin = R.T @ v
        nu = np.concatenate([nu_lin, self.w])
        a_lin = self.f + R.T @ G_NED - np.cross(self.w, nu_lin)
        if self._prev_w is None:
            ang_acc = np.zeros(3)
        else:
            ang_acc = np.zeros(3) if self._dt_tick <= 0 else \
                (self.w - self._prev_w) / self._dt_tick
        h["ok"] = True
        h["meas"] = {"eta": np.array([p[0], p[1], p[2], phi, theta, psi]),
                     "nu": nu,
                     "nudot": np.concatenate([a_lin, ang_acc])}
        h["p_ned"] = p
        h["v_ned"] = v
        h["yaw"] = psi
        return h

    def note_tick(self, dt: float) -> None:
        """Tell the estimator the control period, so the angular-acceleration
        finite difference in ``state()`` divides by the right thing."""
        self._dt_tick = float(dt)
        self._prev_w = self.w.copy()

    _dt_tick = 0.05

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict:
        return {
            "attitude": self.attitude,
            "ahrs_tau_s": self.ahrs_tau_s,
            "accel_trust_m_s2": self.accel_trust,
            "z_source": self.z_source,
            "max_dt_s": self.max_dt_s,
            "static_window_s": self.static_window_s,
            "static_ok": bool(self.static_ok),
            "gyro_ok": bool(self.gyro_ok),
            "static_note": self.static_note or None,
            "accel_static_correction": [float(v) for v in self._c_static],
            "gyro_bias_rad_s": [float(v) for v in self._b_gyro],
            "gyro_bias_source": self.gyro_bias_source,
            "accel_static_source": self.accel_ref_source,
            "static_quality": dict(self.static_quality),
            "samples": int(self.n_samples),
            "rejected": int(self.n_rejected),
            "dropped": int(self.n_dropped),
            "ungated": int(self.n_ungated),
            "calibration": self.calib.meta(),
        }
