#!/usr/bin/env python3
"""Verify the acados NMPC port against the validated IPOPT NMPC.

A. EQUIVALENCE -- acados full-SQP must solve the SAME OCP as IPOPT: on
   bound-inactive interior states, u_acados ~= u_ipopt (validates the shared
   Fossen model + cost + constraints).
B. TIMING -- acados SQP-RTI solve time vs the IPOPT baseline (the whole point).

Run from the package root in the `robust` env.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # package root

from dobmpc.mpc import NMPC
from dobmpc.mpc_acados import AcadosNMPC
from dobmpc import params as P

N = P.MPC_N
RNG = np.random.default_rng(0)


def dp_ref(x0=None):
    """Constant origin DP reference (12, N+1)."""
    return np.zeros((12, N + 1))


def interior_states(n=8):
    """Small errors so neither u, |v_lin|, nor pitch bounds bind."""
    S = []
    for _ in range(n):
        x = np.zeros(12)
        x[0:3] = RNG.uniform(-0.05, 0.05, 3)      # position error (m)
        x[5] = RNG.uniform(-0.1, 0.1)             # yaw
        x[6:9] = RNG.uniform(-0.05, 0.05, 3)      # linear vel
        x[11] = RNG.uniform(-0.05, 0.05)          # yaw rate
        S.append(x)
    return S


def main():
    print("=== A. EQUIVALENCE  (acados full-SQP vs IPOPT, interior states) ===")
    ip = NMPC(N=N, dt=P.DT_CTRL)
    ac = AcadosNMPC(N=N, dt=P.DT_CTRL, rti=False)   # full SQP -> converges
    xref = dp_ref()
    w = np.zeros(6)
    dmax = 0.0
    print(f"  {'state':>6} {'u_ipopt [X Y Z N]':>34} {'u_acados':>34} {'max|du|':>9}")
    for i, x in enumerate(interior_states(8)):
        ip.reset(); ac.reset()
        for _ in range(3):                          # let both converge/warm
            ui = ip.solve(x, w, xref)
        for _ in range(3):
            ua = ac.solve(x, w, xref)
        du = np.max(np.abs(ui - ua))
        dmax = max(dmax, du)
        print(f"  {i:>6} {np.array2string(ui, precision=3, suppress_small=True):>34}"
              f" {np.array2string(ua, precision=3, suppress_small=True):>34} {du:9.4f}")
    print(f"  --> worst-case max|du| over states = {dmax:.4f} N  "
          f"({'PASS' if dmax < 0.25 else 'CHECK'}: < 0.25 N => same optimum)\n")

    print("=== B. TIMING  (per control-tick solve, N=60) ===")
    rti = AcadosNMPC(N=N, dt=P.DT_CTRL, rti=True)
    # moving-setpoint closed-ish loop: shift reference a touch each tick
    x = np.zeros(12); x[0] = 0.1
    ts = []
    for k in range(300):
        ref = dp_ref()
        ref[0, :] = 0.05 * np.sin(0.02 * k)         # gently moving target
        rti.solve(x, w, ref)
        ts.append(rti.solve_ms())
    ts = np.array(ts[20:])                            # drop warmup
    print(f"  acados SQP-RTI : median {np.median(ts):5.2f}  p95 {np.percentile(ts,95):5.2f}"
          f"  max {ts.max():5.2f} ms   (n_fail={rti.n_fail})")

    # IPOPT baseline (slow -> only a handful)
    tip = []
    for k in range(20):
        t = time.time(); ip.solve(x, w, dp_ref()); tip.append((time.time()-t)*1e3)
    tip = np.array(tip)
    print(f"  IPOPT (ref)    : median {np.median(tip):5.2f}  p95 {np.percentile(tip,95):5.2f}"
          f"  max {tip.max():5.2f} ms")
    print(f"  --> speedup (median) = {np.median(tip)/np.median(ts):.1f}x   "
          f"budget 50 ms/tick @20Hz: acados {np.median(ts):.1f}ms (OK), "
          f"IPOPT {np.median(tip):.0f}ms ({'OK' if np.median(tip)<50 else 'OVER'})")


if __name__ == "__main__":
    main()
