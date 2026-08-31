#!/usr/bin/env python3
"""Autonomous square-trajectory mission for the BlueROV2 teleop.

Phase machine: APPROACH the global origin -> (auto-start recording) TRACK a
size×size square in the x,y plane for N laps with a continuously moving setpoint ->
(auto-stop recording, save CSV) DONE, then hold at origin. Uses the baseline
PoseController (with trajectory velocity feed-forward) and the CSV Recorder.

Origin is a CORNER of the square; the loop is CCW:
  (0,0) -> (S,0) -> (S,S) -> (0,S) -> (0,0).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recorder import record_row


class SquareMission:
    def __init__(self, controller, recorder, hydro, bid, size=1.0, laps=10,
                 speed=0.15, depth=0.0, approach_tol=0.05, settle_speed=0.05,
                 approach_timeout=30.0, log_hz=50.0):
        self.c = controller
        self.rec = recorder
        self.hydro = hydro
        self.bid = bid
        self.S = float(size)
        self.laps = int(laps)
        self.speed = float(speed)
        self.depth = float(depth)
        self.tol = float(approach_tol)
        self.settle = float(settle_speed)
        self.timeout = float(approach_timeout)
        self.log_dt = 1.0 / float(log_hz)
        self.P = 4.0 * self.S                       # perimeter (arclength per lap)
        self.phase = "approach"
        self._t0 = None
        self._t_track0 = 0.0
        self._last_log = 0.0
        self.lap = 0
        self.approached = False              # True if it actually settled (vs timed out)
        self.c.set_target((0.0, 0.0, self.depth), yaw_ref=0.0, v_ref=(0.0, 0.0, 0.0))

    def plan_points(self):
        """The planned square as a closed polyline (5,3) at the tracking depth, for
        previewing the trajectory in the viewer/viser before & during the run."""
        S, z = self.S, self.depth
        return np.array([[0, 0, z], [S, 0, z], [S, S, z], [0, S, z], [0, 0, z]], float)

    def _square(self, s):
        """(point, unit tangent) at arclength s in [0, P) around the CCW square."""
        S = self.S
        if s < S:
            return (s, 0.0), (1.0, 0.0)
        if s < 2 * S:
            return (S, s - S), (0.0, 1.0)
        if s < 3 * S:
            return (3 * S - s, S), (-1.0, 0.0)
        return (0.0, 4 * S - s), (0.0, -1.0)

    def step(self, model, data):
        t = data.time
        if self._t0 is None:
            self._t0 = t
        p = np.asarray(data.xpos[self.bid], float)

        if self.phase == "approach":
            self.c.set_target((0.0, 0.0, self.depth), yaw_ref=0.0, v_ref=(0.0, 0.0, 0.0))
            self.c.apply(model, data)
            err = float(np.linalg.norm(p - np.array([0.0, 0.0, self.depth])))
            spd = float(np.linalg.norm(data.qvel[:3]))
            settled = err < self.tol and spd < self.settle
            if settled or (t - self._t0 > self.timeout):
                self.approached = bool(settled)
                self.rec.start()
                self.phase = "track"
                self._t_track0 = t
                self._last_log = t - self.log_dt
                if settled:
                    print(f"[square] reached origin -> recording + tracking "
                          f"{self.laps}-lap {self.S:.2f}m square @ {self.speed:.2f} m/s")
                else:
                    print(f"[square] APPROACH TIMED OUT (err={err:.2f} m) -> recording "
                          f"anyway; {self.laps}-lap square (data starts off-origin)")

        elif self.phase == "track":
            s_tot = self.speed * (t - self._t_track0)
            self.lap = int(s_tot // self.P)
            if self.lap >= self.laps:
                saved = self.rec.stop()
                self.phase = "done"
                print(f"[square] done {self.laps} laps -> saved {saved} "
                      f"({self.rec.n} rows)")
                self.c.set_target((0.0, 0.0, self.depth), yaw_ref=0.0,
                                  v_ref=(0.0, 0.0, 0.0))
                self.c.apply(model, data)
                return
            (x, y), (tx, ty) = self._square(s_tot % self.P)
            self.c.set_target((x, y, self.depth), yaw_ref=0.0,
                              v_ref=(self.speed * tx, self.speed * ty, 0.0))
            self.c.apply(model, data)
            if t - self._last_log >= self.log_dt:
                self.rec.log(record_row(data, self.bid, self.hydro))
                self._last_log = t

        else:  # done -> hold at origin
            self.c.set_target((0.0, 0.0, self.depth), yaw_ref=0.0, v_ref=(0.0, 0.0, 0.0))
            self.c.apply(model, data)

    @property
    def done(self):
        return self.phase == "done"

    def status(self):
        if self.phase == "approach":
            return "approach -> origin"
        tag = "" if self.approached else "  [approach timed out]"
        if self.phase == "track":
            return f"track  lap {self.lap + 1}/{self.laps}  ({self.rec.n} rows){tag}"
        return f"done  ({self.rec.n} rows saved){tag}"

    def close(self):
        """Stop+save any in-progress recording (e.g. on Ctrl-C mid-mission)."""
        return self.rec.stop()
