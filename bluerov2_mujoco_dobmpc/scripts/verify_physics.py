"""First-principles physics checks for the MuJoCo plant.

Unlike validate_plant.py (which checks the MuJoCo port against the analytic
Fossen integrator - both share fossen.py), these tests compare the plant
against HAND-COMPUTABLE physics, so a wrong sign or coefficient inside
fossen.py itself would be caught here.

T1  Coriolis passivity      nu.(C nu) = 0 exactly (skew-symmetry)
T2  Terminal ascent          buoyancy/damping balance, closed-form root
T3  Roll pendulum            overdamped (zeta~15): no oscillation, and the
                             decay matches the slow-pole prediction
T4  Energy monotonicity      free decay: E = KE + potential never increases
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from bluerov2mj import fossen
from bluerov2mj import params as P
from bluerov2mj.mujoco_env import BlueROV2MujocoEnv

NO_NOISE = dict(pos=0, ang=0, lin_vel=0, ang_vel=0, lin_acc=0, ang_acc=0)

# T1 -- Coriolis matrices must be power-neutral: nu.(C nu) = 0
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(1000):
    nu = rng.uniform(-2, 2, 6)
    worst = max(worst, abs(nu @ fossen.coriolis_rb(nu)),
                abs(nu @ fossen.coriolis_added(nu)))
print(f"[T1] max |nu.(C nu)|            = {worst:.2e}   (theory: 0; "
      f"a sign error here pumps energy)")

# T2 -- release at rest, zero control: terminal ascent speed from
#       (B-W) = |Dl_w| w + |Dnl_w| w^2  ->  closed-form root
roots = np.roots([abs(P.DNL[2]), abs(P.DL[2]), -P.NET_BUOYANCY])
w_hand = roots[roots > 0][0]
env = BlueROV2MujocoEnv(meas_noise=NO_NOISE)
env.reset()
for _ in range(400):                     # 20 s >> tau = m_eff/|Dl| ~ 0.5 s
    _, x = env.step(np.zeros(4))
print(f"[T2] terminal ascent: MuJoCo    = {-x[8]*100:.3f} cm/s   "
      f"hand-computed = {w_hand*100:.3f} cm/s   "
      f"(rose {(-20 - x[2])*100:.1f} cm in 20 s)")

# T3 -- roll release from 15 deg: zeta = c / (2 sqrt(I K)) >> 1, so the
#       response must be non-oscillatory and follow the slow pole K/c
zeta = abs(P.DL[3]) / (2 * np.sqrt(P.IX * P.ZG * P.WEIGHT))
lam = P.ZG * P.WEIGHT / abs(P.DL[3])
env.reset(eta0=(0, 0, -20, np.radians(15), 0, 0))
phis = []
for _ in range(400):
    _, x = env.step(np.zeros(4))
    phis.append(x[3])
phis = np.degrees(np.array(phis))
flips = int(np.sum(np.diff(np.sign(phis)) != 0))
phi_pred = 15.0 * np.exp(-lam * 20.0)
print(f"[T3] roll 15deg release: zero-crossings = {flips} (overdamped, "
      f"zeta = {zeta:.1f})   phi(20 s): MuJoCo = {phis[-1]:.2f} deg, "
      f"slow-pole prediction = {phi_pred:.2f} deg")

# T4 -- free decay from a mixed velocity: total mechanical energy
#       E = 0.5 nu.M_total.nu - W z_cg + B z_cb must be non-increasing
def energy(e):
    nu = e._nu_true()
    z_cg = e.data.xipos[e.bid][2]
    z_cb = e.data.xpos[e.bid][2]
    return 0.5 * nu @ (fossen.M_TOTAL @ nu) - P.WEIGHT * z_cg \
        + P.BUOYANCY * z_cb

env.reset(nu0=(0.4, 0.3, 0.2, 0.3, 0.3, 0.4))
E = [energy(env)]
for _ in range(160):                     # 8 s
    env.step(np.zeros(4))
    E.append(energy(env))
E = np.array(E)
dE = np.diff(E)
print(f"[T4] energy free decay: E0 = {E[0]:+.3f} J -> E(8s) = {E[-1]:+.3f} J"
      f"   dissipated = {E[0]-E[-1]:.3f} J   "
      f"max single-step INCREASE = {max(dE.max(), 0):.2e} J")
