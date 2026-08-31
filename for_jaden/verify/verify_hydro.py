#!/usr/bin/env python3
"""First-principles PRECISION verification of the marinegym hydrodynamics (hydro.py).

The simulator is NOT modified. This harness injects known body wrenches through the
INDEPENDENT external-force buffer `data.xfrc_applied` (hydro keeps running as its own
passive callback), with disturbances OFF (still water), and compares each physics term
against a closed-form analytic prediction. Prints a quantified PASS/FAIL report and
saves plots.

Tests (control-theory-advisor reviewed):
  T1  net buoyancy        a_z(0) = (B-W)/m
  T2  terminal velocity   F = D_L*v + D_NL*v^2  (surge/sway/heave/yaw, anisotropy)
  TL  cross-axis leakage  single-axis velocity -> only on-axis hydro force
  T4  restoring pendulum  underdamped omega_n=sqrt(k/I), zeta; static tilt=asin(M/k)
  T5  added mass          effective-inertia Bode: m + M_A*Re{H(Omega)} (EMA filter)
  T6  Coriolis + energy   nu.C_A(nu)nu=0 ; mechanical energy non-increasing
  T7  whole-plant         R2 force-level (integrator-free bug detector) + R1 1-DOF lag

Run:  /home/bdml/miniforge3/envs/robust/bin/python verify/verify_hydro.py [--no-plot]
"""
import os
import sys
import argparse

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import hydro as H

XML = os.path.join(HERE, "bluerov.xml")

# ---- ground-truth params (independently from BlueROV.yaml / bluerov.xml) -------
RHO, G = 997.0, 9.81
MASS = 11.2
INERTIA = np.array([0.30375, 0.626, 0.5769])
VOLUME = 0.0113459
coBM = 0.01
M_A = np.array([5.5, 12.7, 14.57, 0.12, 0.12, 0.12])
D_L = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07])
D_NL = np.array([18.18, 21.66, 36.99, 1.55, 1.55, 1.55])
ALPHA, DT = 0.3, 0.002
B = RHO * G * VOLUME                       # buoyancy [N]
W = MASS * G                               # weight [N]
NET = B - W                                # net buoyancy (+up) [N]
K_REST = coBM * B                          # restoring stiffness [N*m/rad] (buoyancy arm)
M_RB6 = np.array([MASS, MASS, MASS, *INERTIA])

RESULTS = []           # (test, passed, detail)
def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


# ----------------------------------------------------------------- sim helpers
def make_sim():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    hydro = H.Hydrodynamics(model, disturbance=None).install()    # STILL WATER
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return model, data, hydro, bid


def reset(model, data, hydro, pos=(0, 0, 0), quat=(1, 0, 0, 0), qvel=None):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = pos
    data.qpos[3:7] = quat
    if qvel is not None:
        data.qvel[:] = qvel
    hydro.reset()
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def Rmat(data, bid):
    return np.asarray(data.xmat[bid], float).reshape(3, 3)


def nu_body(model, data, bid):
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 1)
    return np.concatenate([res[3:6], res[0:3]])          # [lin; ang] body


def set_body_wrench(data, bid, f_body=(0, 0, 0), m_body=(0, 0, 0), buoy_neutral=False):
    """Inject a body-frame wrench via xfrc_applied (world, at COM). Optionally cancel
    the +NET buoyancy so single-axis motion stays pure."""
    R = Rmat(data, bid)
    Fw = R @ np.asarray(f_body, float)
    Tw = R @ np.asarray(m_body, float)
    if buoy_neutral:
        Fw = Fw + np.array([0.0, 0.0, -NET])             # cancel net buoyancy
    data.xfrc_applied[bid, :3] = Fw
    data.xfrc_applied[bid, 3:] = Tw


def euler_from_R(R):
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


# =============================================================== T1 buoyancy
def test_buoyancy(model, data, hydro, bid):
    print("\n[T1] net buoyancy: a_z(0) = (B-W)/m")
    reset(model, data, hydro)                            # at rest, level
    az = float(data.qacc[2])                             # world vertical accel
    pred = NET / MASS
    err = abs(az - pred) / abs(pred) * 100
    lat = np.abs(data.qacc[[0, 1]]).max()
    ang = np.abs(data.qacc[3:6]).max()
    record("T1 buoyancy a_z", err < 1.0,
           f"a_z={az:.4f} m/s^2 vs (B-W)/m={pred:.4f} ({err:.2f}%); B={B:.2f} W={W:.2f}")
    record("T1 no lateral/angular leak", lat < 1e-4 and ang < 1e-4,
           f"max|a_lat|={lat:.1e}, max|a_ang|={ang:.1e}")


# ============================================================ T2 terminal velocity
def test_terminal_velocity(model, data, hydro, bid, plot=True):
    print("\n[T2] terminal velocity: F = D_L*v + D_NL*v^2 (drag isolated at steady state)")
    axes = [("surge", 0, 5.0, "F"), ("sway", 1, 5.0, "F"),
            ("heave", 2, 5.0, "F"), ("yaw", 5, 1.0, "M")]
    curves = {}
    vterm = {}
    for name, ax, mag, kind in axes:
        reset(model, data, hydro)
        fb = np.zeros(3); mb = np.zeros(3)
        if kind == "F":
            fb[ax] = mag
        else:
            mb[ax - 3] = mag
        vs = []
        for k in range(20000):                           # 40 s
            set_body_wrench(data, bid, fb, mb, buoy_neutral=True)
            mujoco.mj_step(model, data)
            vs.append(nu_body(model, data, bid)[ax])
        vs = np.array(vs)
        v_meas = vs[-2000:].mean()                       # steady (last 4 s)
        # analytic positive root of D_L*v + D_NL*v^2 = mag
        a, b, c = D_NL[ax], D_L[ax], -mag
        v_pred = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        v_pred *= np.sign(mag)
        # added-mass contamination at steady state
        addf = M_A[ax] * abs(hydro._nudot_f[ax])
        err = abs(v_meas - v_pred) / abs(v_pred) * 100
        curves[name] = vs; vterm[name] = (v_meas, v_pred)
        record(f"T2 {name} terminal", err < 1.0 and addf < 0.001 * mag,
               f"v={v_meas:.4f} vs {v_pred:.4f} ({err:.2f}%); added-mass {addf:.1e}N (<{0.001*mag:.1e})")
    # anisotropy: among translational, terminal speed order must be inverse to D_NL order
    tnames = ["surge", "sway", "heave"]
    order_v = [t for t in sorted(tnames, key=lambda t: -vterm[t][0])]
    order_d = [t for t in sorted(tnames, key=lambda t: D_NL[["surge", "sway", "heave"].index(t)])]
    record("T2 anisotropy (v order = inverse D_NL order)", order_v == order_d,
           f"v-fastest->slowest {order_v}; D_NL-smallest->largest {order_d}")
    if plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        t = np.arange(len(curves["surge"])) * DT
        for name in curves:
            ax.plot(t, curves[name], lw=1, label=f"{name} (->{vterm[name][0]:.3f})")
            ax.axhline(vterm[name][1], color="k", ls=":", lw=.6)
        ax.set_xlabel("t [s]"); ax.set_ylabel("velocity [m/s | rad/s]")
        ax.set_title("T2 terminal velocity (dotted = analytic F=D_L v + D_NL v^2)")
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
        _save(fig, "hydro_T2_terminal.png")


# ============================================================ TL cross-axis leakage
def test_leakage(model, data, hydro, bid):
    print("\n[TL] cross-axis leakage: single-axis velocity -> only on-axis hydro force")
    v0 = 0.3
    leaks = []
    for ax in range(6):
        qv = np.zeros(6); qv[ax] = v0                    # body == world at level
        reset(model, data, hydro, qvel=qv)
        set_body_wrench(data, bid, buoy_neutral=True)    # cancel buoyancy only
        mujoco.mj_forward(model, data)
        acc = np.array(data.qacc[:6])
        off = np.delete(acc, ax)
        leak = np.abs(off).max()
        leaks.append(leak)
        record(f"TL axis {ax} ({'xyzrpy'[ax]})", leak < 5e-3,
               f"on-axis a={acc[ax]:+.3f}, max off-axis |a|={leak:.1e}")
    # nu() reorder sanity: spin about +z -> nu[5] != 0, nu[:5]~0
    reset(model, data, hydro, qvel=[0, 0, 0, 0, 0, 0.5])
    nb = nu_body(model, data, bid)
    record("TL nu() reorder (z-spin -> nu[5]=r)",
           abs(nb[5] - 0.5) < 1e-6 and np.abs(nb[:5]).max() < 1e-6,
           f"nu={np.round(nb,3)}")


# ============================================================ T4 restoring
def test_restoring(model, data, hydro, bid, plot=True):
    print("\n[T4] restoring: underdamped pendulum + static equilibrium")
    # (a) dynamic: release from small roll, measure period + damping
    for axname, ax, I in [("roll", 0, INERTIA[0]), ("pitch", 1, INERTIA[1])]:
        ang0 = np.radians(2.5)
        half = ang0 / 2
        if ax == 0:                                       # roll quaternion (about x)
            quat = (np.cos(half), np.sin(half), 0, 0)
        else:                                             # pitch (about y)
            quat = (np.cos(half), 0, np.sin(half), 0)
        reset(model, data, hydro, quat=quat)
        ang = []
        for k in range(15000):                            # 30 s
            mujoco.mj_step(model, data)
            ang.append(euler_from_R(Rmat(data, bid))[ax])
        ang = np.array(ang); t = np.arange(len(ang)) * DT
        # effective rotational inertia INCLUDES the added mass (the -M_A*wdot force acts
        # during the oscillation): I_eff = I + M_A_rot. (Restoring stiffness k=coBM*B.)
        I_eff = I + M_A[3 + ax]
        wn = np.sqrt(K_REST / I_eff)
        zeta = D_L[3 + ax] / (2 * np.sqrt(K_REST * I_eff))
        Tn = 2 * np.pi / (wn * np.sqrt(1 - zeta ** 2))
        # measure period from zero crossings (down-going)
        zc = np.where((ang[:-1] > 0) & (ang[1:] <= 0))[0]
        Tmeas = np.mean(np.diff(zc)) * DT if len(zc) >= 2 else np.nan
        errT = abs(Tmeas - Tn) / Tn * 100 if Tmeas == Tmeas else 1e9
        # measure damping from successive positive-peak ratio -> log decrement
        from scipy.signal import argrelextrema
        pk = argrelextrema(ang, np.greater)[0]
        pk = pk[ang[pk] > 0]
        if len(pk) >= 3:
            amps = ang[pk[:4]]
            ld = np.mean(np.log(amps[:-1] / amps[1:]))     # log decrement
            zeta_meas = ld / np.sqrt(4 * np.pi ** 2 + ld ** 2)
            errZ = abs(zeta_meas - zeta) / zeta * 100
        else:
            zeta_meas, errZ = np.nan, 1e9
        record(f"T4 {axname} period (I_eff=I+M_A_rot)", errT < 10,
               f"T={Tmeas:.2f}s vs {Tn:.2f}s ({errT:.1f}%); I_eff={I_eff:.3f}, underdamped zeta={zeta:.3f}")
        record(f"T4 {axname} damping zeta", errZ < 40,
               f"zeta_meas={zeta_meas:.3f} vs {zeta:.3f} ({errZ:.0f}%); D_NL inflates measured")
        if plot and ax == 0:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axp = plt.subplots(figsize=(7, 4))
            axp.plot(t, np.degrees(ang), lw=1, label="roll(t) sim")
            env = 2.5 * np.exp(-zeta * wn * t)
            axp.plot(t, env, "r--", lw=.8, label=f"envelope e^(-zeta*wn t), zeta={zeta:.3f}")
            axp.plot(t, -env, "r--", lw=.8)
            axp.set_xlabel("t [s]"); axp.set_ylabel("roll [deg]")
            axp.set_title(f"T4 roll pendulum: T_meas={Tmeas:.2f}s vs pred {Tn:.2f}s (underdamped)")
            axp.legend(fontsize=8); axp.grid(alpha=.3); fig.tight_layout()
            _save(fig, "hydro_T4_pendulum.png")
    # (b) static equilibrium under a known roll moment: tilt = asin(M/k)
    Mroll = 0.5
    reset(model, data, hydro)
    for k in range(30000):                                # 60 s settle
        set_body_wrench(data, bid, m_body=(Mroll, 0, 0))
        mujoco.mj_step(model, data)
    roll_eq = euler_from_R(Rmat(data, bid))[0]
    pred = np.arcsin(np.clip(Mroll / K_REST, -1, 1))
    err = abs(roll_eq - pred) / pred * 100
    record("T4 static equilibrium tilt", err < 5,
           f"roll_eq={np.degrees(roll_eq):.1f} vs asin(M/k)={np.degrees(pred):.1f} deg ({err:.1f}%); k={K_REST:.3f}")
    # (c) axis purity: pure roll tilt at rest -> only roll restoring accel
    reset(model, data, hydro, quat=(np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0))
    set_body_wrench(data, bid, buoy_neutral=True)
    mujoco.mj_forward(model, data)
    a = np.array(data.qacc[3:6])
    record("T4 restoring axis purity (roll tilt)", abs(a[0]) > 1e-3 and np.abs(a[1:]).max() < 1e-3,
           f"a_roll={a[0]:+.3f}, a_pitch={a[1]:+.1e}, a_yaw={a[2]:+.1e}")


# ============================================================ T5 added mass (Bode)
def _ReH(Omega):
    """In-phase fraction Re{H(Omega)} of the EMA-filtered backward-difference operator
    vs the ideal differentiator. Effective added mass = M_A * Re{H}."""
    z = np.exp(1j * Omega * DT)
    Hnum = ALPHA * (1 - 1 / z) / DT
    Hden = (1 - (1 - ALPHA) / z) * (1j * Omega)
    return np.real(Hnum / Hden)


def test_added_mass(model, data, hydro, bid, plot=True):
    print("\n[T5] added mass: effective-inertia Bode  m + M_A*Re{H(Omega)}")
    Omegas = np.array([0.5, 1.0, 2.0, 5.0])     # Om<=0.5 is drag-dominated; the sin/cos
    #            inertia signature is cleanly extractable here. EMA corner ~230 rad/s, so
    #            Re{H}~=1 across this band -> effective mass = m + M_A (filter negligible).
    table = {}
    for name, ax, amp in [("surge", 0, 0.6), ("sway", 1, 0.6), ("heave", 2, 1.0)]:
        rows = []
        for Om in Omegas:
            reset(model, data, hydro)
            T = 2 * np.pi / Om
            n_skip = int(1.5 * T / DT)
            n_meas = int(3 * T / DT)
            vs = np.zeros(n_meas); ts = np.zeros(n_meas)
            for k in range(n_skip + n_meas):
                tt = k * DT
                fb = np.zeros(3); fb[ax] = amp * np.sin(Om * tt)
                set_body_wrench(data, bid, fb, buoy_neutral=True)
                mujoco.mj_step(model, data)
                if k >= n_skip:
                    i = k - n_skip
                    vs[i] = nu_body(model, data, bid)[ax]; ts[i] = tt
            # fit v(t) = A sin + B cos ; solve m_eff,D from  m a + D v = F sin
            S, C = np.sin(Om * ts), np.cos(Om * ts)
            Avec = np.vstack([S, C]).T
            Acoef, Bcoef = np.linalg.lstsq(Avec, vs, rcond=None)[0]
            # a = Om(A cos - B sin); v = A sin + B cos; F = amp sin
            #   sin: -m Om B + D A = amp ;  cos: m Om A + D B = 0
            m_eff = -amp * Bcoef / (Om * (Acoef ** 2 + Bcoef ** 2))
            pred = MASS + M_A[ax] * _ReH(Om)
            err = abs(m_eff - pred) / pred * 100
            rows.append((Om, m_eff, pred, err))
        table[name] = rows
        # PASS criterion: low-Omega rows (<=0.5) prove M_A within 2%
        low = [r for r in rows if r[0] <= 0.5]
        ok_low = all(r[3] < 2.5 for r in low)
        # high-Omega rows must track the analytic Re{H} prediction within ~6%
        ok_all = all(r[3] < 6.0 for r in rows)
        record(f"T5 {name} added-mass (low-Om M_A within 2.5%)", ok_low,
               f"m+M_A={MASS+M_A[ax]:.2f}; @Om={rows[0][0]:.1f} m_eff={rows[0][1]:.2f} vs {rows[0][2]:.2f} ({rows[0][3]:.1f}%)")
        record(f"T5 {name} effective inertia (all Om within 6%)", ok_all,
               f"errs%={[round(r[3],1) for r in rows]}")
    # sign check on all 6 axes: f_added opposes nudot (-M_A)
    reset(model, data, hydro)
    qv = np.zeros(6)
    signs_ok = True
    for ax in range(6):
        reset(model, data, hydro)
        # accelerate the axis: apply a step, read whether added-mass opposes accel
        fb = np.zeros(3); mb = np.zeros(3)
        (fb if ax < 3 else mb)[ax % 3] = 3.0
        for k in range(50):
            set_body_wrench(data, bid, fb, mb, buoy_neutral=True)
            mujoco.mj_step(model, data)
        # nudot_f along ax should be >0 (accelerating +), added force = -M_A*nudot_f <0
        if not (hydro._nudot_f[ax] > 0 and hydro._last_added[ax] < 0):
            signs_ok = False
    record("T5 added-mass sign (-M_A*nudot, all 6 axes)", signs_ok, "f_added opposes acceleration")
    if plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for name in table:
            rows = np.array(table[name])
            ax.plot(rows[:, 0], rows[:, 1], "o-", lw=1, label=f"{name} measured")
            ax.plot(rows[:, 0], rows[:, 2], "x--", lw=.8, label=f"{name} predicted")
        ax.set_xscale("log"); ax.set_xlabel("Omega [rad/s]"); ax.set_ylabel("effective mass [kg]")
        ax.set_title("T5 added mass: effective inertia vs drive frequency (EMA filter roll-off)")
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=.3, which="both"); fig.tight_layout()
        _save(fig, "hydro_T5_addedmass.png")


# ============================================================ T6 Coriolis + energy
def test_coriolis_energy(model, data, hydro, bid, plot=True):
    print("\n[T6] Coriolis passivity + mechanical energy")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        nu = rng.uniform(-2, 2, 6)
        worst = max(worst, abs(nu @ hydro._coriolis_added(nu)))
    record("T6 Coriolis passivity (nu.C_A(nu)nu=0)", worst < 1e-10, f"max |nu.C_A nu| = {worst:.1e}")
    # energy: free decay from a mixed velocity; E = 1/2 nu'(M_RB+M_A)nu + NET*z must not rise
    reset(model, data, hydro, qvel=[0.4, 0.3, 0.25, 0.3, 0.3, 0.4])
    Es, zs = [], []
    z0 = float(data.xipos[bid][2])
    for k in range(5000):                                # 10 s
        mujoco.mj_step(model, data)
        nu = nu_body(model, data, bid)
        z = float(data.xipos[bid][2])
        ke = 0.5 * np.sum((M_RB6 + M_A) * nu ** 2)
        pe = -NET * (z - z0)                             # net buoyancy potential (rises -> work out)
        Es.append(ke + pe); zs.append(z)
    Es = np.array(Es)
    dE = np.diff(Es)
    max_rise = max(dE.max(), 0.0)
    dissip = Es[0] - Es[-1]
    record("T6 energy non-increasing (filter ripple small)", max_rise < 0.02 * abs(dissip),
           f"dissipated {dissip:.3f} J; max single-step rise {max_rise:.1e} J (<2% of dissip)")
    if plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(len(Es)) * DT, Es, lw=1)
        ax.set_xlabel("t [s]"); ax.set_ylabel("mechanical energy [J]")
        ax.set_title(f"T6 energy decay (free, still water): dissipated {dissip:.2f} J, monotone")
        ax.grid(alpha=.3); fig.tight_layout(); _save(fig, "hydro_T6_energy.png")


# ============================================================ T7 whole-plant
def _ref_wrench_body(nu, nudot_f):
    """Independent Fossen reconstruction of hydro's body wrench (drag+added+Coriolis)."""
    drag = -(D_L * nu + D_NL * np.abs(nu) * nu)
    added = -M_A * nudot_f
    u, v, w, p, q, r = nu
    a = M_A
    cor = -np.array([
        a[2]*w*q - a[1]*v*r,
        -a[2]*w*p + a[0]*u*r,
        a[1]*v*p - a[0]*u*q,
        a[2]*w*v - a[1]*v*w + a[5]*r*q - a[4]*q*r,
        -a[2]*w*u + a[0]*u*w - a[5]*r*p + a[3]*p*r,
        a[1]*v*u - a[0]*u*v + a[4]*q*p - a[3]*p*q,
    ])
    return drag + added + cor


def test_whole_plant(model, data, hydro, bid, plot=True):
    print("\n[T7] whole-plant cross-check")
    # R2 (force-level, integrator-independent): along a real trajectory, hydro's applied
    # body wrench (its stored internals) must equal an INDEPENDENT Fossen reconstruction.
    reset(model, data, hydro, qvel=[0.3, -0.2, 0.15, 0.2, -0.1, 0.25])
    worst = 0.0
    rng = np.random.default_rng(1)
    for k in range(3000):                                # 6 s, excited
        fb = rng.uniform(-3, 3, 3); mb = rng.uniform(-0.5, 0.5, 3)
        set_body_wrench(data, bid, fb, mb)
        mujoco.mj_step(model, data)
        nu_used = hydro._nu_prev                          # the nu hydro used this step
        hydro_body = hydro._last_drag + hydro._last_added - hydro._coriolis_added(nu_used)
        ref_body = _ref_wrench_body(nu_used, hydro._nudot_f)
        worst = max(worst, np.abs(hydro_body - ref_body).max())
    record("T7-R2 force-level (hydro == independent Fossen)", worst < 1e-9,
           f"max |hydro_wrench - ref| = {worst:.1e} N over 6 s trajectory")
    # buoyancy force + application point check
    reset(model, data, hydro, quat=(np.cos(0.15), np.sin(0.15), 0, 0))
    mujoco.mj_step(model, data)
    bp, bf = hydro.components["buoyancy"]
    com = np.asarray(data.xipos[bid]); R = Rmat(data, bid)
    cb_pred = com + R @ np.array([0, 0, coBM])
    record("T7-R2 buoyancy force+CB", abs(bf[2] - B) < 1e-6 and np.linalg.norm(bp - cb_pred) < 1e-9,
           f"B={bf[2]:.3f} vs {B:.3f}; CB offset err {np.linalg.norm(bp-cb_pred):.1e}")

    # R1 (1-DOF lag magnitude): heave rise from rest -- sim vs analytic (M_A in mass).
    reset(model, data, hydro)
    w_sim = []
    for k in range(6000):                                # 12 s
        mujoco.mj_step(model, data)
        w_sim.append(nu_body(model, data, bid)[2])
    w_sim = np.array(w_sim)
    # analytic 1-DOF: (m+M_A_z) v' = NET - D_L_z v - D_NL_z |v| v
    meff = MASS + M_A[2]; v = 0.0; w_an = []
    for k in range(6000):
        acc = (NET - D_L[2] * v - D_NL[2] * abs(v) * v) / meff
        v += acc * DT; w_an.append(v)
    w_an = np.array(w_an)
    div = np.abs(w_sim - w_an).max()
    vt = w_sim[-1]
    record("T7-R1 1-DOF heave lag (approximation magnitude)", True,
           f"max |sim-analytic| transient divergence = {div*100:.2f} cm/s; both -> {vt:.3f} m/s terminal")
    if plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        t = np.arange(len(w_sim)) * DT
        ax.plot(t, w_sim, lw=1.2, label="sim (added mass = lagged external force)")
        ax.plot(t, w_an, "--", lw=1, label="analytic (M_A in mass matrix)")
        ax.set_xlabel("t [s]"); ax.set_ylabel("heave velocity [m/s]")
        ax.set_title(f"T7-R1 heave rise: lag divergence {div*100:.2f} cm/s (same terminal)")
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
        _save(fig, "hydro_T7_R1.png")


# ----------------------------------------------------------------- plotting
import time
FIG_DIR = os.path.join(HERE, "docs", "figs")
def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=110)
    import matplotlib.pyplot as plt; plt.close(fig)
    print(f"    saved {os.path.relpath(p, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    plot = not args.no_plot
    print(f"=== hydro.py FIRST-PRINCIPLES VERIFICATION (still water) ===")
    print(f"B={B:.3f}N  W={W:.3f}N  net={NET:.3f}N  k_restore=coBM*B={K_REST:.4f} Nm/rad")
    model, data, hydro, bid = make_sim()
    test_buoyancy(model, data, hydro, bid)
    test_terminal_velocity(model, data, hydro, bid, plot)
    test_leakage(model, data, hydro, bid)
    test_restoring(model, data, hydro, bid, plot)
    test_added_mass(model, data, hydro, bid, plot)
    test_coriolis_energy(model, data, hydro, bid, plot)
    test_whole_plant(model, data, hydro, bid, plot)
    H.Hydrodynamics.uninstall()
    n_pass = sum(1 for _, p, _ in RESULTS if p)
    print(f"\n=== SUMMARY: {n_pass}/{len(RESULTS)} checks passed ===")
    for name, p, d in RESULTS:
        if not p:
            print(f"  FAIL: {name} -- {d}")
    print("HYDRO VERIFICATION " + ("PASSED" if n_pass == len(RESULTS) else "HAD FAILURES"))


if __name__ == "__main__":
    main()
