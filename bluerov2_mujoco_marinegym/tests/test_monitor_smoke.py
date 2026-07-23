#!/usr/bin/env python3
"""Headless smoke test for monitor.py (run with QT_QPA_PLATFORM=offscreen).

Validates the dashboard widget builds + redraws on synthetic samples (2D guaranteed;
3D degrades gracefully offscreen), and that the spawn-process round-trip
(push + close) never raises and shuts down cleanly.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import math
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: F401
import monitor as M
from pyqtgraph.Qt import QtWidgets


def _sample(i):
    t = i * 0.033
    vt = (0.25 * math.sin(t), 0.12 * math.cos(0.7 * t), 0.06 * math.sin(2.0 * t))
    pos = (0.01 * i, 0.2 * math.sin(0.2 * t), -0.1 * math.cos(0.2 * t))
    return {"t": t, "cur": (0.2, 0.0, 0.0), "wav": vt, "vtot": vt, "pos": pos, "dist": True}


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # 1. widget builds + redraws on ~200 synthetic samples
    d = M.MonitorDashboard(window_s=5.0)
    for i in range(200):
        d.add_sample(_sample(i))
    d.redraw()
    vx_x, vx_y = d.c_vx.getData()
    assert vx_x is not None and len(vx_x) > 0, "velocity curve empty"
    assert len(d.sc_xy.data) > 0, "XY trajectory scatter empty"
    # zero-disturbance arrow must not blow up
    d.add_sample({"t": 6.7, "cur": (0,)*3, "wav": (0,)*3, "vtot": (0.0, 0.0, 0.0),
                  "pos": (2.0, 0.0, 0.0), "dist": False})
    d.redraw()
    print(f"[1] widget OK  gl_ok={d.gl_ok}  vel_pts={len(vx_x)}  xy_pts={len(d.sc_xy.data)}")

    # 2. spawn-process round-trip: push + clean close
    h = M.MonitorHandle(window_s=5.0)
    for i in range(60):
        h.push(_sample(i))
        time.sleep(0.004)
    assert h._alive, "handle died unexpectedly during push"
    h.close()
    assert not h._alive, "handle not closed"
    print("[2] process round-trip OK  (push never raised, child joined)")

    # 3. push after close is a safe no-op
    h.push(_sample(0))
    print("[3] post-close push is a no-op OK")

    # 4. degenerate-time guards: all-equal timestamps + backward jump (reset)
    d2 = M.MonitorDashboard(window_s=5.0)
    for _ in range(5):                                  # all-equal t -> norm must be 0, no NaN
        d2.add_sample({"t": 7.0, "vtot": (0.1, 0.0, 0.0), "pos": (1.0, 0.0, 0.0)})
    d2.redraw()
    xy = d2.sc_xy.data
    assert len(xy) > 0, "equal-ts case produced no points"
    d2.add_sample({"t": 0.0, "vtot": (0.0, 0.1, 0.0), "pos": (0.0, 0.0, 0.0)})  # reset jump
    assert len(d2.t) == 1, "backward time jump did not clear ring buffers"
    d2.add_sample({"t": 0.1, "vtot": (0.0, 0.1, 0.0), "pos": (0.0, 0.1, 0.0)})
    d2.redraw()
    print("[4] degenerate-time guards OK (equal-ts + reset jump)")

    print("\nMONITOR SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
