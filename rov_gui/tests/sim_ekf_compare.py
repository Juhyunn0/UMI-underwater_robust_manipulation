#!/usr/bin/env python3
"""sim_ekf_compare.py — the state assembler vs the robot_localization EKF on
one synthetic sensor stream.

SYNTHETIC. Nothing here is a measurement of any device, pool or vehicle: the
motion is PRESCRIBED (open loop — no controller reacts to either estimate)
and every sensor number below is an assumption. What it does exercise is the
real code of both estimators with realistic TIMING — four sensors at their
own rates, each late by its own latency, a 20 Hz tick asynchronous to all —
and it scores what each one hands the controller AT THE TICK against the
truth at that same instant.

    assumed  tags   15 Hz, 0.13 s late (+U(0, 20 ms)), 0.8 mm, yaw 0.3 deg
             IMU    25 Hz ATTITUDE (+ the latest SCALED_IMU2), 20 ms late,
                    stamped on ARRIVAL (as the station does), roll/pitch
                    0.1 deg, gyro 3 mrad/s, accel 0.03 m/s^2
             depth  10 Hz, 50 ms late (arrival-stamped), 3 mm
             DVL    12 Hz, 0.19 s late (t_valid exact), 3 mm/s, lever arm
                    0.2 m forward
             tick   20 Hz
    (tag / DVL numbers = sim_dvl_monitors.NOMINAL, so the two tools agree)

Estimators (all fed the same events in arrival order):
    asm          StateAssembler, tag finite difference (every run to date)
    asm+dvl      StateAssembler + DvlAid IN (mount measured, opt-out of the
                 proof so it is IN from the start) — the 2026-10-01 design
    ekf          RlEkfNode, r_l defaults (no lagged-data smoothing), DVL in
    ekf-nodvl    the same without the DVL
    ekf-smooth   smooth_lagged_data true, history_length 1 s (r_l's own fix)

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/sim_ekf_compare.py [seeds]

Columns: horizontal position error RMS / p95 (mm), body (u, v) error RMS
(cm/s), yaw error RMS (deg), heave rate w error RMS (cm/s).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import dvl as DV                              # noqa: E402
from rov_gui.control import rl_ekf as EK                           # noqa: E402
from rov_gui.control.state_assembler import (StateAssembler,       # noqa: E402
                                             rot_zyx)
from rov_gui.state import Conn, DvlSample, NavFix, VehicleImu      # noqa: E402

G = 9.80665
NOMINAL = {"tag_hz": 15.0, "tag_lat": 0.13, "tag_sig": 0.0008,
           "tag_yaw_sig_deg": 0.3,
           "imu_hz": 25.0, "imu_lat": 0.02, "rp_sig_deg": 0.1,
           "gyro_sig": 0.003, "acc_sig": 0.03, "acc_bias": (0.0, 0.0, 0.0),
           "baro_hz": 10.0, "baro_lat": 0.05, "baro_sig": 0.003,
           "dvl_hz": 12.0, "dvl_lat": 0.19, "dvl_sig": 0.003}
ARM = np.array([0.2, 0.0, 0.0])            # DVL lever arm, body FRD (true)
ESTIMATORS = ("asm", "asm+dvl", "ekf", "ekf-nodvl", "ekf-smooth")


def _euler_rates_to_body(phi, th, dphi, dth, dpsi):
    """ZYX Euler rates -> body rates (p, q, r)."""
    p = dphi - dpsi * math.sin(th)
    q = dth * math.cos(phi) + dpsi * math.cos(th) * math.sin(phi)
    r = -dth * math.sin(phi) + dpsi * math.cos(th) * math.cos(phi)
    return p, q, r


def truth(profile, T, dt_f=0.002, tilt=True):
    """Integrate ``profile(t) -> (u, v, r)`` (surge, sway in the LEVEL
    heading frame, yaw rate) plus a small roll/pitch/heave wobble."""
    n = int(T / dt_f) + 1
    tt = np.arange(n) * dt_f
    uvr = np.array([profile(t) for t in tt])
    yaw = np.cumsum(uvr[:, 2]) * dt_f
    if tilt:
        phi = math.radians(3.0) * np.sin(2 * math.pi * 0.2 * tt)
        th = math.radians(2.0) * np.sin(2 * math.pi * 0.13 * tt + 0.7)
        zw = 0.02 * np.sin(2 * math.pi * 0.1 * tt)
    else:
        phi = th = zw = np.zeros(n)
    vx = uvr[:, 0] * np.cos(yaw) - uvr[:, 1] * np.sin(yaw)
    vy = uvr[:, 0] * np.sin(yaw) + uvr[:, 1] * np.cos(yaw)
    vz = np.gradient(zw, dt_f)
    px, py = np.cumsum(vx) * dt_f, np.cumsum(vy) * dt_f
    pz = 0.5 + zw
    ax, ay, az = (np.gradient(v, dt_f) for v in (vx, vy, vz))
    dphi, dth = np.gradient(phi, dt_f), np.gradient(th, dt_f)
    return {"t": tt, "px": px, "py": py, "pz": pz, "vx": vx, "vy": vy,
            "vz": vz, "ax": ax, "ay": ay, "az": az, "phi": phi, "th": th,
            "yaw": yaw, "dphi": dphi, "dth": dth, "dpsi": uvr[:, 2]}


def _at(tr, key, t):
    return float(np.interp(t, tr["t"], tr[key]))


def _state_at(tr, t):
    phi, th, psi = (_at(tr, k, t) for k in ("phi", "th", "yaw"))
    R = rot_zyx(phi, th, psi)
    vw = np.array([_at(tr, k, t) for k in ("vx", "vy", "vz")])
    aw = np.array([_at(tr, k, t) for k in ("ax", "ay", "az")])
    pqr = np.array(_euler_rates_to_body(phi, th, _at(tr, "dphi", t),
                                        _at(tr, "dth", t),
                                        _at(tr, "dpsi", t)))
    p = np.array([_at(tr, k, t) for k in ("px", "py", "pz")])
    return p, (phi, th, psi), R, vw, aw, pqr


def events(tr, T, seed, c):
    """Every sensor message as (arrival, kind, payload), arrival-sorted."""
    rng = np.random.default_rng(seed)
    ev = []
    t = 0.0
    while t < T - 0.5:
        p, (phi, th, psi), R, *_ = _state_at(tr, t)
        psi_n = psi + math.radians(rng.normal(0, c["tag_yaw_sig_deg"]))
        Rn = rot_zyx(phi, th, psi_n)
        pn = p + rng.normal(0, c["tag_sig"], 3)
        ev.append((t + c["tag_lat"] + rng.uniform(0, 0.02), "fix", NavFix(
            t_capture=t, n_tags=2, p_ned=tuple(pn),
            R_ned_body=tuple(Rn.ravel()), yaw_ned=psi_n,
            conn=Conn.ONLINE, stamp=t)))
        t += 1.0 / c["tag_hz"]
    bias = np.asarray(c["acc_bias"], float)
    t = 0.004
    while t < T - 0.5:
        p, (phi, th, psi), R, vw, aw, pqr = _state_at(tr, t)
        f = R.T @ (aw - np.array([0.0, 0.0, G]))
        f = f + bias + rng.normal(0, c["acc_sig"], 3)
        g = pqr + rng.normal(0, c["gyro_sig"], 3)
        rp = np.radians(rng.normal(0, c["rp_sig_deg"], 2))
        ta = t + c["imu_lat"]
        ev.append((ta, "imu", {"roll": phi + rp[0], "pitch": th + rp[1],
                               "yaw": psi + 0.3, "p": g[0], "q": g[1],
                               "r": g[2], "ax": f[0], "ay": f[1],
                               "az": f[2], "t": ta}))
        t += 1.0 / c["imu_hz"]
    t = 0.011
    while t < T - 0.5:
        p, *_ = _state_at(tr, t)
        ev.append((t + c["baro_lat"], "baro",
                   {"depth": p[2] - 0.5 + 0.3 + rng.normal(0, c["baro_sig"]),
                    "t": t + c["baro_lat"]}))
        t += 1.0 / c["baro_hz"]
    t = 0.017
    while t < T - 0.5:
        p, ang, R, vw, aw, pqr = _state_at(tr, t)
        nu = R.T @ vw
        v_pt = nu + np.cross(pqr, ARM)
        v_dev = v_pt + rng.normal(0, c["dvl_sig"], 3)
        ev.append((t + c["dvl_lat"] + rng.uniform(0, 0.01), "dvl", DvlSample(
            t_valid=t, t_rx=t + c["dvl_lat"], v_dvl=tuple(v_dev), valid=True,
            fom=0.003, altitude=0.5, n_beams=4,
            covariance=tuple((np.eye(3) * c["dvl_sig"] ** 2).ravel()),
            conn=Conn.ONLINE, stamp=t)))
        t += 1.0 / c["dvl_hz"]
    ev.sort(key=lambda e: e[0])
    return ev


class _Imu:
    """The station's VehicleImu snapshot: the latest ATTITUDE/SCALED_IMU2
    and the latest baro, each with its own host stamp."""

    def __init__(self):
        self.v = None
        self.depth = None
        self.t_baro = None

    def snap(self):
        if self.v is None:
            return None
        d = self.v
        return VehicleImu(roll=d["roll"], pitch=d["pitch"], yaw=d["yaw"],
                          p=d["p"], q=d["q"], r=d["r"], ax=d["ax"],
                          ay=d["ay"], az=d["az"], depth_m=self.depth,
                          t_att=d["t"], t_imu=d["t"], t_baro=self.t_baro,
                          conn=Conn.ONLINE, stamp=d["t"])


def run(profile, T, seed, which=ESTIMATORS, tilt=True, **over):
    c = dict(NOMINAL)
    c.update(over)
    tr = truth(profile, T, tilt=tilt)
    ev = events(tr, T, seed, c)
    dvl_cfg = {"mount_rpy_deg": [0.0, 0.0, 0.0],
               "mount_t_flu": [float(ARM[0]), 0.0, 0.0],
               "require_proven": False}
    mount = DV.mount_from_cfg(DV.resolve(dvl_cfg))
    est = {}
    for name in which:
        if name.startswith("asm"):
            asm = StateAssembler(z_source="pressure", vel_lp_alpha=0.6,
                                 nudot_source="imu", tag_stale_s=0.7)
            aid = None
            if name == "asm+dvl":
                aid = DV.DvlAid(dvl_cfg, fd_lp_gain=0.6 / math.sqrt(1.4))
            est[name] = {"asm": asm, "aid": aid, "err": []}
        else:
            cfg = {}
            if name == "ekf-smooth":
                cfg = {"smooth_lagged_data": True, "history_length_s": 1.0}
            node = EK.RlEkfNode(cfg, dvl_mount=mount, z_source="pressure")
            est[name] = {"node": node, "dvl": name != "ekf-nodvl",
                         "err": []}
    imu = _Imu()
    fix = None
    gyro_hist = []
    i = 0
    tick = 0.3
    t_score = 5.0                    # let every filter settle first
    while tick < T - 0.6:
        while i < len(ev) and ev[i][0] <= tick:
            arr, kind, pl = ev[i]
            i += 1
            if kind == "fix":
                fix = pl
                for name, e in est.items():
                    if "node" in e:
                        e["node"].feed_tag(fix, arr)
                    elif e["aid"] is not None:
                        e["aid"].note_fix(fix, imu.snap())
            elif kind == "imu":
                imu.v = pl
                gyro_hist.append((pl["t"], pl["p"], pl["q"], pl["r"]))
                for e in est.values():
                    if "node" in e:
                        e["node"].feed_imu(imu.snap(), arr)
            elif kind == "baro":
                imu.depth, imu.t_baro = pl["depth"], pl["t"]
                for name, e in est.items():
                    if "node" in e:
                        off = est_offset(est)
                        if off is not None:
                            e["node"].feed_depth(pl["depth"] + off, pl["t"],
                                                 arr)
            elif kind == "dvl":
                om = _omega_at(gyro_hist, pl.t_valid)
                for e in est.values():
                    if "node" in e:
                        if e["dvl"]:
                            e["node"].feed_dvl(pl, arr)
                    elif e["aid"] is not None:
                        e["aid"].note_sample(pl, om)
        snap = imu.snap()
        p_t, ang_t, R_t, vw_t, _, pqr_t = _state_at(tr, tick)
        nu_t = R_t.T @ vw_t
        for name, e in est.items():
            if "node" in e:
                e["node"].advance(tick)
                out = e["node"].output()
                meas = EK.to_meas(out) if out is not None else None
            else:
                if fix is None or snap is None:
                    continue
                vel = disp = None
                aid = e["aid"]
                if aid is not None:
                    if not aid.use and aid.sample is not None:
                        aid.set_use(True)
                    v, _why = aid.velocity(tick)
                    if v is not None:
                        vel = v[:2]
                        d3 = aid.displacement(float(fix.t_capture), tick)
                        disp = None if d3 is None else d3[:2]
                meas, h = e["asm"].step(fix, snap, tick, 0.05, vel_xy=vel,
                                        disp_xy=disp)
                if aid is not None:
                    aid.note_tag_velocity(h.get("nu_fd") if meas else None,
                                          tick, tag_age=h.get("tag_age"))
            if meas is None or tick < t_score:
                continue
            eta, nu = meas["eta"], meas["nu"]
            e["err"].append((
                math.hypot(eta[0] - p_t[0], eta[1] - p_t[1]),
                math.hypot(nu[0] - nu_t[0], nu[1] - nu_t[1]),
                abs(math.degrees(math.atan2(math.sin(eta[5] - ang_t[2]),
                                            math.cos(eta[5] - ang_t[2])))),
                abs(nu[2] - nu_t[2]),
                abs(eta[2] - p_t[2])))
        tick += 0.05
    res = {}
    for name, e in est.items():
        a = np.asarray(e["err"]) if e["err"] else np.full((1, 5), np.nan)
        res[name] = {"pos_rms_mm": 1e3 * math.sqrt(np.mean(a[:, 0] ** 2)),
                     "pos_p95_mm": 1e3 * float(np.percentile(a[:, 0], 95)),
                     "vel_rms_cm_s": 1e2 * math.sqrt(np.mean(a[:, 1] ** 2)),
                     "yaw_rms_deg": math.sqrt(np.mean(a[:, 2] ** 2)),
                     "w_rms_cm_s": 1e2 * math.sqrt(np.mean(a[:, 3] ** 2)),
                     "z_rms_mm": 1e3 * math.sqrt(np.mean(a[:, 4] ** 2)),
                     "n": len(e["err"])}
        if "node" in e:
            res[name]["meta"] = e["node"].meta()
    return res


def est_offset(est):
    """The pressure-z offset the assembler anchored (the worker hands the
    EKF the SAME offset, so both estimators sit at one depth)."""
    for e in est.values():
        if "asm" in e and e["asm"].z_offset is not None:
            return e["asm"].z_offset
    # no assembler in this run: the synthetic baro reads depth - 0.5 + 0.3
    return 0.2


def _omega_at(hist, t):
    """Gyro at time t from the arrival-stamped history (as the worker's
    _omega_at does): linear interpolation, None outside it."""
    if not hist or t < hist[0][0] - 0.1:
        return None
    for k in range(len(hist) - 1, -1, -1):
        if hist[k][0] <= t:
            if k + 1 < len(hist):
                a, b = hist[k], hist[k + 1]
                w = (t - a[0]) / max(b[0] - a[0], 1e-9)
                return tuple((1 - w) * a[j] + w * b[j] for j in (1, 2, 3))
            return hist[k][1:]
    return hist[0][1:]


# ------------------------------------------------------------------ profiles
def station(t):
    return (0.012 * math.sin(0.9 * t) + 0.008 * math.sin(2.3 * t + 1.0),
            0.010 * math.sin(0.7 * t + 0.5) + 0.006 * math.sin(3.1 * t), 0.0)


def _smooth_box(x, lo, hi, ramp):
    """1 inside [lo, hi), 0 outside, with cosine edges ``ramp`` wide. The
    IMU is in the loop here, so a velocity STEP (sim_dvl_monitors' square)
    would hand the accelerometer a 2 ms spike of tens of m/s^2 — a
    simulator artefact, not a vehicle."""
    def edge(z):
        if z <= 0.0:
            return 0.0
        if z >= ramp:
            return 1.0
        return 0.5 - 0.5 * math.cos(math.pi * z / ramp)
    return edge(x - lo) * (1.0 - edge(x - hi))


def square(v, ramp=0.5):
    """sim_dvl_monitors' square (4 s legs) with ``ramp``-second edges."""
    def f(t):
        ph = t % 16.0
        u = v * (_smooth_box(ph, 0, 4, ramp) - _smooth_box(ph, 8, 12, ramp))
        s = v * (_smooth_box(ph, 4, 8, ramp) - _smooth_box(ph, 12, 16, ramp))
        return (u, s, 0.0)
    return f


def passes(t):                               # 0.3 m/s trapezoids, 0.15 m/s^2
    ph = t % 12.0
    a = 0.15
    if ph < 2:
        u = a * ph
    elif ph < 4:
        u = 0.3
    elif ph < 8:
        u = 0.3 - a * (ph - 4)
    elif ph < 10:
        u = -0.3
    else:
        u = -0.3 + a * (ph - 10)
    return (u, 0.5 * u * math.sin(0.5 * t), 0.0)


def yawing(t):                               # station hold + yaw sweeps
    return (0.01 * math.sin(t), 0.01 * math.cos(0.8 * t),
            0.3 * math.sin(2 * math.pi * t / 6.0))


def kicks(t):                                # a 0.2 m/s push every 8 s
    ph = t % 8.0                             # (0.3 s rise, 0.8 s decay)
    if ph < 0.3:
        u = 0.2 * (0.5 - 0.5 * math.cos(math.pi * ph / 0.3))
    elif ph < 4.0:
        u = 0.2 * math.exp(-(ph - 0.3) / 0.8)
    else:
        u = 0.0
    return (u, -0.5 * u, 0.0)


PROFILES = (("station", station), ("square 0.1", square(0.1)),
            ("passes 0.3", passes), ("yawing", yawing), ("kicks", kicks))

# What moves each answer: one assumption changed at a time, on `passes`.
SENSITIVITY = (
    ("nominal", {}),
    ("tag 0.25 s late", {"tag_lat": 0.25}),
    ("tag 0.05 s late", {"tag_lat": 0.05}),
    ("accel bias 0.1", {"acc_bias": (0.1, 0.0, 0.0)}),
    ("DVL 0.05 s late", {"dvl_lat": 0.05}),
    ("IMU 50 Hz", {"imu_hz": 50.0}),
)


def main(seeds=(1, 2, 3), T=60.0):
    t0 = time.time()
    print(__doc__.split("\n\n")[0])
    print(f"seeds {list(seeds)}, {T:.0f} s per run, scored from t = 5 s\n")
    hdr = (f"{'profile':12s} {'estimator':11s} {'pos rms':>8s} "
           f"{'pos p95':>8s} {'uv rms':>8s} {'yaw rms':>8s} {'w rms':>7s}"
           f" {'z rms':>6s}   tag late (mean/max s)")
    for label, prof in PROFILES:
        print(hdr)
        acc = {n: [] for n in ESTIMATORS}
        lates = {n: [] for n in ESTIMATORS}
        for s in seeds:
            r = run(prof, T, s)
            for n in ESTIMATORS:
                acc[n].append(r[n])
                m = r[n].get("meta")
                if m:
                    lates[n].append((m["late_mean_s"]["pose0"] or 0.0,
                                     m["late_max_s"]["pose0"]))
        for n in ESTIMATORS:
            k = acc[n]

            def mean(key):
                return float(np.mean([x[key] for x in k]))
            late = ""
            if lates[n]:
                late = (f"{np.mean([a for a, _ in lates[n]]):.3f} / "
                        f"{max(b for _, b in lates[n]):.3f}")
            print(f"{label:12s} {n:11s} {mean('pos_rms_mm'):7.1f}mm "
                  f"{mean('pos_p95_mm'):7.1f}mm "
                  f"{mean('vel_rms_cm_s'):6.2f}cm/s "
                  f"{mean('yaw_rms_deg'):6.2f}° "
                  f"{mean('w_rms_cm_s'):5.2f}cm/s {mean('z_rms_mm'):5.1f}mm"
                  f"   {late}")
        print()
    print("sensitivity on `passes 0.3` (one assumption changed per row)")
    print(f"{'case':17s} {'estimator':11s} {'pos rms':>8s} {'uv rms':>10s}"
          f" {'yaw rms':>8s}")
    for label, over in SENSITIVITY:
        acc = {n: [] for n in ESTIMATORS}
        for s in seeds:
            r = run(passes, T, s, **over)
            for n in ESTIMATORS:
                acc[n].append(r[n])
        for n in ESTIMATORS:
            k = acc[n]
            print(f"{label:17s} {n:11s} "
                  f"{np.mean([x['pos_rms_mm'] for x in k]):7.1f}mm "
                  f"{np.mean([x['vel_rms_cm_s'] for x in k]):7.2f}cm/s "
                  f"{np.mean([x['yaw_rms_deg'] for x in k]):6.2f}°")
        print()
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    seeds = tuple(int(a) for a in sys.argv[1:]) or (1, 2, 3)
    main(seeds)
