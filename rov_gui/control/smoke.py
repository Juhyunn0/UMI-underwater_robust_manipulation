#!/usr/bin/env python3
"""smoke.py — the P0 gate: does the sim controller stack run in THIS env?

    ROV_MODEL is taken from config/hw_mpc.yaml (or --rov-model).

    ~/miniforge3/envs/rovgui-pose/bin/python -m rov_gui.control.smoke
    ... --check-map        also verify the floor tag map loads + spot distances
    ... --solves 200       longer timing run

Exit codes: 0 = acados RTI real-time OK; 1 = anything else (IPOPT-only,
build failure, EAOB divergence). The GUI refuses to engage in exactly the
same conditions — this tool exists so that verdict arrives on the bench, not
at the pool.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mpc-config", default="config/hw_mpc.yaml")
    ap.add_argument("--nav-config", default="config/hw_nav.yaml")
    ap.add_argument("--rov-model", default=None)
    ap.add_argument("--solves", type=int, default=100)
    ap.add_argument("--check-map", action="store_true")
    a = ap.parse_args(argv)

    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")

    from rov_gui.control.geometry import MpcConfig, NavConfig

    cfg = MpcConfig.load(a.mpc_config)
    if a.rov_model:
        cfg.rov_model = a.rov_model
    print(f"rov_model={cfg.rov_model} mode={cfg.mode} ctrl_hz={cfg.ctrl_hz}")

    rc = 0
    if a.check_map:
        rc |= _check_map(NavConfig.load(a.nav_config))

    # ---- controller build + probe (the acados-under-numpy2 question)
    from rov_gui.control.mpc_bridge import HwDobMpc

    t0 = time.perf_counter()
    try:
        ctrl = HwDobMpc("dobmpc", cfg, log=lambda m: print(f"  {m}"))
    except Exception as e:                                       # noqa: BLE001
        print(f"FAIL: controller build: {type(e).__name__}: {e}")
        return 1
    print(f"controller built in {time.perf_counter() - t0:.1f}s "
          f"(solver={ctrl.solver_kind})")
    if ctrl.solver_kind != "acados":
        print("FAIL: acados did not load — running on IPOPT (~83 ms/solve) "
              "is not real-time. Check ACADOS_SOURCE_DIR=/home/bdml/acados "
              "and the _acados_env preload.")
        return 1

    # ---- timing over a plausible closed-loop-ish sequence
    rng = np.random.default_rng(0)
    eta = np.array([-2.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    nu = np.zeros(6)
    ctrl.set_target_ned(eta[:3], 0.0)
    times, statuses = [], []
    for k in range(int(a.solves)):
        nudot = rng.normal(0.0, 0.3, 6)
        t0 = time.perf_counter()
        u, info = ctrl.step(eta + rng.normal(0, 0.005, 6),
                            nu + rng.normal(0, 0.01, 6), nudot,
                            k * 0.05)
        times.append(1e3 * (time.perf_counter() - t0))
        statuses.append(info["status"])
        if not np.isfinite(u).all():
            print(f"FAIL: non-finite u at step {k}")
            return 1
    times = np.array(times)
    p50, p99 = np.percentile(times, [50, 99])
    print(f"{len(times)} step() calls (EAOB+solve): "
          f"p50 {p50:.2f} ms  p99 {p99:.2f} ms  max {times.max():.2f} ms  "
          f"n_fail={ctrl.n_fail}")
    if p99 > 45.0:
        print("FAIL: p99 above the 45 ms tick budget")
        return 1

    # ---- EAOB sanity: stationary, zero wrench -> w_hat must stay bounded
    w_norm = float(np.linalg.norm(ctrl.w_hat))
    print(f"EAOB after {a.solves} noisy stationary ticks: |w_hat| = "
          f"{w_norm:.2f} (bounded by clip {list(ctrl.w_clip)})")
    if not np.isfinite(w_norm):
        print("FAIL: EAOB diverged")
        return 1

    print("OK: acados RTI real-time in this environment.")
    return rc


def _check_map(nav) -> int:
    from rov_gui.control.tagnav import TagMap

    m = TagMap.load(nav.tag_map_path)
    print(f"tag map {nav.tag_map_path}: {len(m)} tags, anchor {m.anchor_id}")
    ids = sorted(m.poses)
    bad = 0
    for i, j in zip(ids[:-1], ids[1:]):
        d = float(np.linalg.norm(m.poses[i][1] - m.poses[j][1]))
        if not (0.01 < d < 6.0):
            print(f"  suspicious distance {i}-{j}: {d:.3f} m")
            bad += 1
    print(f"  adjacent-id spot distances: {len(ids) - 1} checked, {bad} "
          f"suspicious. Tape-measure at least 6 pairs against these numbers "
          f"before trusting the map (P1).")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
