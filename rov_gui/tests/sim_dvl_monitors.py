#!/usr/bin/env python3
"""sim_dvl_monitors.py — do the DVL's runtime monitors trip when they should?

SYNTHETIC. Nothing here is a measurement of any device, pool or vehicle: the
motion is PRESCRIBED (open loop — no controller reacts to a wrong velocity),
and every sensor number below is an assumption. What it does exercise is the
real ``DvlAid`` + ``StateAssembler`` code with realistic TIMING — two sensors
at different rates, each arriving late by its own latency, a control tick
asynchronous to both — which the unit tests in test_dvl.py do not.

    assumed  tags 15 Hz, 0.13 s late, 0.8 mm position noise
             DVL  12 Hz, 0.19 s late, 3 mm/s noise, 0.2 m lever arm
             tick 20 Hz

Questions, one table each:

    A1-A3  a CORRECT mount: does it ever trip? (trips are latched, so one
           false trip costs the DVL for the session) — nominal, with tag
           jumps, across latencies
    A4     the same under conditions NOT expected to be clean: wrong
           capture stamps, noisy or slow tags, a slow DVL, a wrong lever arm
    A5     a DVL that glitches (once or repeatedly) or is simply noisy
           (inside fom_max): no trip expected, only holds
    A5b    a false lock that LASTS: a trip expected past ~0.5-1 s
    A6     a mount the proof ACCEPTS at the edge of its tolerance, at speed
    B      a WRONG mount flown under the opt-out (require_proven false):
           how long from IN to the trip?
    C      a correct mount under the opt-out, creeping near the monitors'
           speed floor for minutes: any false trip?

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/sim_dvl_monitors.py [seeds]

The output of the run the docs quote is kept beside the DVL fixture:
rov_gui/tests/fixtures/sim_dvl_monitors_20261002.txt
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
from rov_gui.control.state_assembler import (StateAssembler,       # noqa: E402
                                             rot_zyx)
from rov_gui.state import Conn, DvlSample, NavFix, VehicleImu      # noqa: E402

NOMINAL = {"tag_hz": 15.0, "tag_lat": 0.13, "tag_sig": 0.0008,
           "dvl_hz": 12.0, "dvl_lat": 0.19, "dvl_sig": 0.003}
ARM = np.array([0.2, 0.0, 0.0])            # the DVL's true lever arm, body FRD


def run(profile, T, seed, mount_err=None, opt_out=False, prove_first=True,
        jumps=(), doubles=(), stamp_bias=0.0, arm_err=0.0, glitch=None,
        fom=0.003, noise_after=None, **over):
    """One flight. ``profile(t) -> (u, v, r)``: body surge, sway (m/s) and
    yaw rate (rad/s), the TRUTH.

    mount_err   2x2: the device reads ``mount_err @ v`` (None = right mount)
    jumps       [(t, dx, dy)] the tag solution steps by this and stays
    doubles     [t] a second solver's fix 1 ms after the first, 4 cm off
    stamp_bias  seconds added to every fix's capture stamp (a wrong stamp)
    arm_err     metres added to the CONFIGURED lever arm's x (a wrong tape)
    glitch      (every_s, n_reports, dv[, t_start[, axis]]): from t_start
                (default 20 s) on, every ``every_s`` seconds ``n_reports``
                consecutive DVL reports read ``dv`` m/s more than is true
                along ``axis`` (0 surge, 1 sway = default) — a false lock
                off a wall; every_s 1000 = once
    fom         what the device reports as its own accuracy
    noise_after (t, sigma): the DVL noise becomes ``sigma`` from ``t`` on
    """
    c = dict(NOMINAL)
    c.update(over)
    rng = np.random.default_rng(seed)
    dt_f = 0.002
    n = int(T / dt_f) + 1
    tt = np.arange(n) * dt_f
    uvr = np.array([profile(t) for t in tt])
    yaw = np.cumsum(uvr[:, 2]) * dt_f
    vx = uvr[:, 0] * np.cos(yaw) - uvr[:, 1] * np.sin(yaw)
    vy = uvr[:, 0] * np.sin(yaw) + uvr[:, 1] * np.cos(yaw)
    px, py = np.cumsum(vx) * dt_f, np.cumsum(vy) * dt_f

    def at(arr, t):
        return float(np.interp(t, tt, arr))

    def imu_at(t):
        return VehicleImu(roll=0.0, pitch=0.0, yaw=at(yaw, t), p=0.0, q=0.0,
                          r=at(uvr[:, 2], t), ax=0.0, ay=0.0, az=-9.80665,
                          depth_m=0.5, t_att=t, t_imu=t, t_baro=t,
                          conn=Conn.ONLINE, stamp=t)

    # selection legacy (2026-10-16): this simulation times the SHIPPED
    # ladder's trips, and hw_mpc.yaml / the READMEs quote its fixture.
    cfg = {"mount_rpy_deg": [0.0, 0.0, 0.0],
           "mount_t_flu": [float(ARM[0]) + arm_err, 0.0, 0.0],
           "selection": "legacy"}
    if opt_out:
        cfg["require_proven"] = False
    aid = DV.DvlAid(cfg, fd_lp_gain=0.6 / math.sqrt(1.4))
    asm = StateAssembler(z_source="tag", vel_lp_alpha=0.6, nudot_source="imu",
                         tag_stale_s=0.5)
    M = np.eye(2) if mount_err is None else np.asarray(mount_err, float)

    events = []                              # (arrival, kind, payload)
    doubles = sorted(doubles)
    t = 0.0
    while t < T - c["tag_lat"] - 0.05:
        off = np.zeros(2)
        for tj, dx, dy in jumps:
            if t >= tj:
                off += (dx, dy)
        p = (at(px, t) + off[0] + rng.normal(0, c["tag_sig"]),
             at(py, t) + off[1] + rng.normal(0, c["tag_sig"]), 0.5)
        R = rot_zyx(0.0, 0.0, at(yaw, t))
        arrive = t + c["tag_lat"] + rng.uniform(0, 0.02)
        events.append((arrive, "fix", NavFix(
            t_capture=t + stamp_bias, n_tags=2, p_ned=p,
            R_ned_body=tuple(R.ravel()), yaw_ned=at(yaw, t),
            conn=Conn.ONLINE, stamp=t)))
        if doubles and t >= doubles[0]:
            doubles.pop(0)
            events.append((arrive + 0.0005, "fix", NavFix(
                t_capture=t + stamp_bias + 0.001, n_tags=1,
                p_ned=(p[0] + 0.04, p[1] - 0.03, 0.5),
                R_ned_body=tuple(R.ravel()), yaw_ned=at(yaw, t),
                conn=Conn.ONLINE, stamp=t)))
        t += 1.0 / c["tag_hz"]
    t = 0.017
    g_left = 0
    g_next = float(glitch[3]) if glitch is not None and len(glitch) > 3 \
        else 20.0
    g_axis = int(glitch[4]) if glitch is not None and len(glitch) > 4 else 1
    while t < T - c["dvl_lat"] - 0.05:
        u, v, r = (at(uvr[:, k], t) for k in range(3))
        v_pt = np.array([u, v, 0.0]) + np.cross([0.0, 0.0, r], ARM)
        sig = c["dvl_sig"]
        if noise_after is not None and t >= float(noise_after[0]):
            sig = float(noise_after[1])
        v_dev = np.array([*(M @ v_pt[:2]), 0.0]) + rng.normal(0, sig, 3)
        if glitch is not None:
            if t >= g_next:
                g_left, g_next = int(glitch[1]), g_next + float(glitch[0])
            if g_left > 0:
                g_left -= 1
                v_dev[g_axis] += float(glitch[2])
        events.append((t + c["dvl_lat"] + rng.uniform(0, 0.01), "dvl", (
            DvlSample(t_valid=t, t_rx=t + c["dvl_lat"], v_dvl=tuple(v_dev),
                      valid=True, fom=float(fom), altitude=0.5, n_beams=4,
                      conn=Conn.ONLINE, stamp=t), (0.0, 0.0, r))))
        t += 1.0 / c["dvl_hz"]
    events.sort(key=lambda e: e[0])

    out = {"t_proven": None, "t_trip": None, "trip": "", "n_ticks": 0,
           "n_dvl": 0, "why": {}, "armed_t": None, "lag_s": None}
    fix = None
    i = 0
    tick = 0.3
    while tick < T:
        while i < len(events) and events[i][0] <= tick:
            _, kind, pl = events[i]
            i += 1
            if kind == "fix":
                fix = pl
                aid.note_fix(fix, imu_at(tick))
            else:
                aid.note_sample(pl[0], pl[1])
        if aid.proven and out["t_proven"] is None:
            out["t_proven"] = tick
            out["lag_s"] = aid.proof.get("lag_s")
        if not aid.use and aid.sample is not None and (
                opt_out or (aid.proven if prove_first else True)):
            if aid.set_use(True) == "":
                out["armed_t"] = tick
        if aid.tripped and out["t_trip"] is None:
            out["t_trip"], out["trip"] = tick, aid.tripped
        if fix is not None:
            v, why = aid.velocity(tick)
            vel = disp = None
            if v is not None:
                vel = v[:2]
                d3 = aid.displacement(float(fix.t_capture), tick)
                disp = None if d3 is None else d3[:2]
            meas, h = asm.step(fix, imu_at(tick), tick, 0.05, vel_xy=vel,
                               disp_xy=disp)
            if meas is not None:
                aid.note_tag_velocity(h["nu_fd"], tick, tag_age=h["tag_age"])
                if aid.use:
                    out["n_ticks"] += 1
                    if h["vel_src"] == "dvl":
                        out["n_dvl"] += 1
                    else:
                        k = why.split(" (")[0]
                        out["why"][k] = out["why"].get(k, 0) + 1
            else:
                aid.note_tag_velocity(None, tick)
        tick += 0.05
    out["n_gross"] = aid.n_gross
    out["n_innov"] = aid.n_innovation
    return out


# ------------------------------------------------------------------ profiles
def prove_then(body):
    """15 s of forward-back then left-right passes, then ``body``."""
    def f(t):
        if t < 7.5:
            return (0.12 * math.sin(2 * math.pi * t / 7.5), 0.0, 0.0)
        if t < 15.0:
            return (0.0, 0.10 * math.sin(2 * math.pi * (t - 7.5) / 7.5), 0.0)
        return body(t - 15.0)
    return f


def station(t):
    return (0.012 * math.sin(0.9 * t) + 0.008 * math.sin(2.3 * t + 1.0),
            0.010 * math.sin(0.7 * t + 0.5) + 0.006 * math.sin(3.1 * t), 0.0)


def square(v):
    def f(t):
        ph = (t % 16.0) / 4.0
        u = v if ph < 1 else (-v if 2 <= ph < 3 else 0.0)
        s = v if 1 <= ph < 2 else (-v if ph >= 3 else 0.0)
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


def yawing(t):                               # a station hold + yaw sweeps
    return (0.01 * math.sin(t), 0.01 * math.cos(0.8 * t),
            0.3 * math.sin(2 * math.pi * t / 6.0))


def osc(f_hz, amp, sway=0.6):
    w = 2 * math.pi * f_hz
    return lambda t: (amp * math.sin(w * t),
                      sway * amp * math.sin(w * t + 1.0), 0.0)


def reversals(v):                            # instant reversals, every 3 s
    return lambda t: (v if (t % 6.0) < 3.0 else -v, 0.0, 0.0)


def const(u, v):
    return lambda t: (u, v, 0.0)


# -------------------------------------------------------------------- tables
def false_trips(name, rs):
    trips = [r for r in rs if r["t_trip"] is not None]
    proven = sum(1 for r in rs if r["t_proven"] is not None)
    on = [r["n_dvl"] / max(1, r["n_ticks"]) for r in rs]
    why = {}
    for r in rs:
        for k, v in r["why"].items():
            why[k] = why.get(k, 0) + v
    line = (f"  {name:38s} false trips {len(trips)}/{len(rs)}"
            f"   proven {proven}/{len(rs)}"
            f"   flew on DVL {100 * float(np.mean(on)):5.1f} %"
            f"   gross pairs {sum(r['n_gross'] for r in rs)}")
    if why:
        line += f"   held back {why}"
    print(line, flush=True)
    for r in trips[:2]:
        print(f"      trip at {r['t_trip']:.1f} s: {r['trip'][:100]}")
    return len(trips)


def main(argv) -> int:
    seeds = int(argv[1]) if len(argv) > 1 else 12
    t0 = time.time()
    total = 0
    print("sim_dvl_monitors.py — SYNTHETIC, prescribed motion, assumed "
          f"sensor numbers {NOMINAL}; {seeds} seeds")

    bodies = (("station keeping (1-2 cm/s)", station),
              ("5 cm/s square", square(0.05)),
              ("10 cm/s square", square(0.10)),
              ("0.3 m/s passes, 0.15 m/s^2", passes),
              ("yaw sweeps 0.3 rad/s", yawing),
              ("oscillation 0.5 Hz 6 cm/s", osc(0.5, 0.06)),
              ("oscillation 1 Hz 6 cm/s", osc(1.0, 0.06)),
              ("oscillation 1.5 Hz 4 cm/s", osc(1.5, 0.04)),
              ("reversals 8 cm/s", reversals(0.08)))

    print("\nA1. correct mount, proven by a 15 s pass, then 60 s of:")
    for name, body in bodies:
        total += false_trips(name, [run(prove_then(body), 75.0, s)
                                    for s in range(seeds)])

    print("\nA2. the same with TAG JUMPS (4 cm steps every 7 s, +10 cm at "
          "40 s, a 1 ms double fix every 11 s):")
    jumps = [(20.0 + 7.0 * k, 0.04 * (-1) ** k, 0.03) for k in range(7)] \
        + [(40.0, 0.10, -0.05)]
    for name, body in bodies[:4]:
        total += false_trips(name, [
            run(prove_then(body), 75.0, s, jumps=jumps,
                doubles=[22.0, 33.0, 44.0, 55.0, 66.0])
            for s in range(seeds)])

    print("\nA3. the same across latencies (tag s, DVL s) — false trips "
          "summed over the nine motions above:")
    for tag_lat, dvl_lat in ((0.05, 0.35), (0.25, 0.05), (0.13, 0.05),
                             (0.05, 0.19), (0.30, 0.30)):
        k = sum(1 for _name, body in bodies for s in range(max(2, seeds // 3))
                if run(prove_then(body), 75.0, s, tag_lat=tag_lat,
                       dvl_lat=dvl_lat)["t_trip"] is not None)
        total += k
        print(f"  tag {tag_lat:.2f} s, DVL {dvl_lat:.2f} s:  false trips "
              f"{k}/{len(bodies) * max(2, seeds // 3)}", flush=True)

    print("\nA4. harder conditions (false trips / runs):")
    hard = (("tag noise 2 mm", {"tag_sig": 0.002}),
            ("tag noise 3 mm, tags 10 Hz", {"tag_sig": 0.003, "tag_hz": 10.0}),
            ("tags 5 Hz", {"tag_hz": 5.0}),
            ("DVL 5 Hz, noise 1 cm/s", {"dvl_hz": 5.0, "dvl_sig": 0.01}),
            ("capture stamps 50 ms late", {"stamp_bias": 0.05}),
            ("capture stamps 50 ms early", {"stamp_bias": -0.05}),
            ("capture stamps 100 ms late", {"stamp_bias": 0.10}),
            ("capture stamps 150 ms late", {"stamp_bias": 0.15}),
            ("lever arm 3 cm wrong", {"arm_err": 0.03}),
            ("lever arm 10 cm wrong", {"arm_err": 0.10}))
    for label, kw in hard:
        rs = [(name, run(prove_then(body), 75.0, s, **kw))
              for name, body in bodies for s in range(max(2, seeds // 3))]
        bad = sorted({name for name, r in rs if r["t_trip"] is not None})
        unproven = sum(1 for _n, r in rs if r["t_proven"] is None)
        lags = [r["lag_s"] for _n, r in rs if r["lag_s"] is not None]
        print(f"  {label:28s} {sum(1 for _n, r in rs if r['t_trip'] is not None)}"
              f"/{len(rs)}"
              + (f"   proof's lag_s {1e3 * float(np.median(lags)):+4.0f} ms"
                 if lags else "")
              + (f"   (never proven: {unproven})" if unproven else "")
              + (f"   tripped in: {bad}" if bad else ""), flush=True)

    print("\nA5. a DVL that glitches or is noisy (inside fom_max), correct "
          "mount, 15 s proving pass, then 60 s — trips / runs, and the share "
          "of ticks flown on the DVL (the rest are held back, no latch):")
    few = max(2, seeds // 2)
    rows = (
        ("once: 4 reports -14 cm/s surge at 30 s", {"glitch": (1000.0, 4, -0.14, 30.0, 0)}, {}),
        ("once: 4 reports +14 cm/s surge at 30 s", {"glitch": (1000.0, 4, 0.14, 30.0, 0)}, {}),
        ("once: 4 reports -14 cm/s surge at 22 s", {"glitch": (1000.0, 4, -0.14, 22.0, 0)}, {}),
        ("once: 4 reports +30 cm/s surge, tags 3 mm", {"glitch": (1000.0, 4, 0.30, 25.0, 0)}, {"tag_sig": 0.003}),
        ("once: 3 reports +30 cm/s surge, tags 3 mm", {"glitch": (1000.0, 3, 0.30, 25.0, 0)}, {"tag_sig": 0.003}),
        ("once: 4 reports +25 cm/s sway, tags 3 mm", {"glitch": (1000.0, 4, 0.25, 25.0, 1)}, {"tag_sig": 0.003}),
        ("every 10 s: 1 report +15 cm/s sway", {"glitch": (10.0, 1, 0.15)}, {}),
        ("every 10 s: 2 reports +12 cm/s sway", {"glitch": (10.0, 2, 0.12)}, {}),
        ("every 10 s: 4 reports +14 cm/s sway", {"glitch": (10.0, 4, 0.14)}, {}),
        ("every 10 s: 4 reports -14 cm/s surge", {"glitch": (10.0, 4, -0.14, 20.0, 0)}, {}),
        ("every 5 s: 4 reports +14 cm/s surge", {"glitch": (5.0, 4, 0.14, 20.0, 0)}, {}),
        ("every 3 s: 4 reports +12 cm/s surge", {"glitch": (3.0, 4, 0.12, 20.0, 0)}, {}),
        ("every 3 s: 2 reports +20 cm/s sway", {"glitch": (3.0, 2, 0.20)}, {}),
        ("DVL noise 1 cm/s (fom 0.01)", {"fom": 0.01}, {"dvl_sig": 0.01}),
        ("DVL noise 2 cm/s (fom 0.02)", {"fom": 0.02}, {"dvl_sig": 0.02}),
        ("DVL noise 2.5 cm/s (fom 0.025)", {"fom": 0.025}, {"dvl_sig": 0.025}),
        ("DVL noise 2.5 cm/s from 20 s", {"fom": 0.025, "noise_after": (20.0, 0.025)}, {}),
        ("DVL noise 3 cm/s (fom 0.03)", {"fom": 0.03}, {"dvl_sig": 0.03}),
        ("DVL noise 5 cm/s (fom 0.05)", {"fom": 0.05}, {"dvl_sig": 0.05}),
        # round 5: a glitch's innovation hold must release with tags late
        # enough that no frame is ever "fresh"
        ("once: 3 reports +18 cm/s sway, tags 0.28 s late",
         {"glitch": (1000.0, 3, 0.18, 30.0, 1)}, {"tag_lat": 0.28}),
        ("once: 3 reports +18 cm/s sway, tags 0.35 s late",
         {"glitch": (1000.0, 3, 0.18, 30.0, 1)}, {"tag_lat": 0.35}))
    for label, kw, over in rows:
        row = []
        for name, body in (("station", station),
                           ("5 cm/s square", square(0.05)),
                           ("10 cm/s square", square(0.10)),
                           ("0.3 m/s passes", passes)):
            rs = [run(prove_then(body), 75.0, s, **kw, **over)
                  for s in range(few)]
            k = sum(1 for r in rs if r["t_trip"] is not None)
            pv = sum(1 for r in rs if r["t_proven"] is not None)
            on = 100 * float(np.mean([r["n_dvl"] / max(1, r["n_ticks"])
                                      for r in rs]))
            total += k
            row.append(f"{name}: {k}/{len(rs)}"
                       + (f" (proven {pv})" if pv < len(rs) else "")
                       + f" {on:.0f}%"
                       + (f" [{next(r['trip'] for r in rs if r['t_trip'])[:50]}]"
                          if k else ""))
        print(f"  {label:44s} " + " | ".join(row), flush=True)

    print("\nA5b. a false lock that LASTS (reports reading sway that is "
          "not there, once at 20 s / every 10 s), correct mount — these ARE "
          "meant to trip from about half a second (moving) to a second (at "
          "rest); the rows show where the line falls (trips / runs):")
    for n_rep, dv in ((4, 0.10), (6, 0.10), (6, 0.14), (8, 0.10), (12, 0.10),
                      (12, 0.14), (24, 0.10)):
        row = []
        for every, tag in ((1000.0, "once"), (10.0, "every 10 s")):
            for name, body in (("station", station),
                               ("5 cm/s square", square(0.05))):
                rs = [run(prove_then(body), 75.0, s,
                          glitch=(every, n_rep, dv)) for s in range(few)]
                k = sum(1 for r in rs if r["t_trip"] is not None)
                row.append(f"{tag}, {name}: {k}/{len(rs)}")
        print(f"  {n_rep:2d} reports ({n_rep / NOMINAL['dvl_hz']:.2f} s) "
              f"+{dv * 100:.0f} cm/s   " + " | ".join(row), flush=True)

    print("\nA6. a mount the proof ACCEPTS at the edge of its tolerance, "
          "then straight passes (trips / runs):")

    def cruise(vmax):
        def f(t):
            ph = t % 16.0
            u = vmax * min(1.0, ph / 2.0) if ph < 6 else (
                vmax * max(0.0, 1.0 - (ph - 6) / 2.0) if ph < 8 else (
                    -vmax * min(1.0, (ph - 8) / 2.0) if ph < 14 else
                    -vmax * max(0.0, 1.0 - (ph - 14) / 2.0)))
            return (u, 0.0, 0.0)
        return f

    def long_prove(body):                    # 45 s of passes, then the body
        base = prove_then(lambda t: (0.0, 0.0, 0.0))
        return lambda t: base(t % 15.0) if t < 45.0 else body(t - 45.0)

    for label, Mx in (("12 deg yaw", np.array(
            [[math.cos(math.radians(12)), math.sin(math.radians(12))],
             [-math.sin(math.radians(12)), math.cos(math.radians(12))]])),
            ("reads 1.2x", np.diag([1.2, 1.2])),
            ("5 deg yaw", np.array(
                [[math.cos(math.radians(5)), math.sin(math.radians(5))],
                 [-math.sin(math.radians(5)), math.cos(math.radians(5))]]))):
        row = []
        for vmax in (0.2, 0.3, 0.4, 0.5):
            rs = [run(long_prove(cruise(vmax)), 95.0, s, mount_err=Mx)
                  for s in range(max(2, seeds // 3))]
            pv = sum(1 for r in rs if r["t_proven"] is not None)
            k = sum(1 for r in rs if r["t_trip"] is not None)
            row.append(f"{vmax:.1f} m/s: {k}/{len(rs)}"
                       + ("" if pv == len(rs) else f" (proven {pv})"))
        print(f"  {label:12s} " + " | ".join(row), flush=True)

    print("\nB. WRONG mount under the opt-out — median seconds from IN to "
          "the trip (6 seeds; NEVER = not within 30 s, with the share of "
          "ticks flown on the wrong DVL):")
    th = math.radians(30)
    wrong = (("surge reversed", np.diag([-1.0, 1.0])),
             ("sway reversed", np.diag([1.0, -1.0])),
             ("90 deg", np.array([[0.0, 1.0], [-1.0, 0.0]])),
             ("30 deg", np.array([[math.cos(th), math.sin(th)],
                                  [-math.sin(th), math.cos(th)]])),
             ("reads 2x", np.diag([2.0, 2.0])))
    w05 = 2 * math.pi * 0.5
    motions = (("station", station),
               ("5+2 cm/s", const(0.05, 0.02)), ("10+4 cm/s", const(0.10, 0.04)),
               ("20+8 cm/s", const(0.20, 0.08)),
               ("osc 3 cm/s", osc(4 / (2 * math.pi), 0.03, 0.4)),
               ("osc 6 cm/s", osc(4 / (2 * math.pi), 0.06, 0.4)),
               ("osc 10 cm/s", osc(4 / (2 * math.pi), 0.10, 0.4)),
               # a straight line 10 and 14 degrees off the bow: one axis
               # moves well above the floor, IN PHASE with the other
               ("20+3.5 line", const(0.20, 0.035)),
               ("30+7.5 0.5Hz", lambda t: (0.30 * math.sin(w05 * t),
                                           0.075 * math.sin(w05 * t), 0.0)))
    def table_b(**kw):
        """Cell: median time to the trip; or, when not every seed trips,
        how many did and the share of ticks flown on the (wrong) DVL."""
        print("  " + " " * 16 + " | ".join(f"{m:>13s}" for m, _ in motions))
        for mname, Mx in wrong:
            row = []
            for _label, prof in motions:
                ts, on = [], []
                for s in range(6):
                    r = run(prof, 30.0, s, mount_err=Mx, opt_out=True,
                            prove_first=False, **kw)
                    ts.append(None if r["t_trip"] is None
                              else r["t_trip"] - (r["armed_t"] or 0.0))
                    on.append(r["n_dvl"] / max(1, r["n_ticks"]))
                ok = [x for x in ts if x is not None]
                pc = f"{100 * float(np.mean(on)):.0f}%"
                row.append(f"{np.median(ok):.1f} s" if len(ok) == len(ts)
                           else f"NEVER {pc}" if not ok
                           else f"{len(ok)}/6 {pc}")
            print(f"  {mname:15s} " + " | ".join(f"{x:>13s}" for x in row),
                  flush=True)

    table_b()
    print("\nB2. the same with a DVL noise of 2 cm/s (fom 0.02): the "
          "percentage is the share of ticks flown on the wrong DVL — the "
          "fit, once it refutes, holds the DVL back even where nothing "
          "trips:")
    table_b(dvl_sig=0.02, fom=0.02)

    print("\nC. correct mount under the opt-out, 300 s near the speed floor "
          "(false trips / runs):")
    for name, body in (("station keeping (1-2 cm/s)", station),
                       ("creep 0.3 Hz, 3.5 cm/s peak", osc(0.3, 0.035, 0.0)),
                       ("creep 0.3 Hz, 5 cm/s peak, both axes",
                        osc(0.3, 0.05, 0.7)),
                       ("3 cm/s square", square(0.03))):
        for label, kw in (("0.8 mm tags", {}), ("3 mm tags",
                                                 {"tag_sig": 0.003})):
            rs = [run(body, 300.0, s, opt_out=True, prove_first=False, **kw)
                  for s in range(max(2, seeds // 3))]
            k = sum(1 for r in rs if r["t_trip"] is not None)
            total += k
            print(f"  {name:38s} {label:12s} {k}/{len(rs)}"
                  + (f"   {next(r['trip'] for r in rs if r['t_trip'])[:80]}"
                     if k else ""), flush=True)

    print(f"\nfalse trips of a correct mount in A1-A3, A5 and C: {total}"
          f"   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
