#!/usr/bin/env python3
"""CSV recorder for the BlueROV2 teleop — logs per-frame disturbance + ROV state +
thruster input. Driven by teleop.py's viser Record / Stop buttons.

One row per logged frame (FLU, z-up world frame): time; global position; orientation
(quaternion + roll/pitch/yaw); velocity (world linear, body angular); disturbance
(current + wave water velocity, kick force, enabled flag); thruster forces u0..u5 (N).
"""
import csv
import os
import threading
import time

import numpy as np

RECORD_FIELDS = [
    "t",                                   # sim time [s]
    "px", "py", "pz",                      # world position [m]
    "qw", "qx", "qy", "qz",                # orientation quaternion (wxyz)
    "roll", "pitch", "yaw",                # euler [rad] (convenience)
    "vx", "vy", "vz",                      # world-frame linear velocity [m/s]
    "wx", "wy", "wz",                      # body-frame angular velocity [rad/s]
    "cur_x", "cur_y", "cur_z",             # current water velocity [m/s]
    "wav_x", "wav_y", "wav_z",             # wave water velocity [m/s]
    "kick_x", "kick_y", "kick_z",          # kick external force [N]
    "u0", "u1", "u2", "u3", "u4", "u5",    # thruster forces written to actuators [N]
    "dist_on",                             # disturbances enabled (0/1)
]


def record_row(data, bid, hydro):
    """Build one CSV row (dict over RECORD_FIELDS) from the live sim state. READ-ONLY."""
    R = np.asarray(data.xmat[bid], float).reshape(3, 3)
    p = np.asarray(data.xpos[bid], float)
    q = np.asarray(data.xquat[bid], float)             # wxyz
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    v = np.asarray(data.qvel[:3], float)               # world linear velocity
    w = np.asarray(data.qvel[3:6], float)              # body angular velocity
    if hydro is not None and getattr(hydro, "water", None):
        cur = np.asarray(hydro.water["current"][1], float)
        wav = np.asarray(hydro.water["wave"][1], float)
        kick = (np.asarray(hydro.components["kick"][1], float)
                if hydro.components else np.zeros(3))
        dist_on = int(bool(hydro.disturbance is not None and hydro.disturbance.enabled))
    else:
        cur = wav = kick = np.zeros(3)
        dist_on = 0
    u = np.asarray(data.ctrl[:6], float)
    vals = ([float(data.time)] + list(p) + list(q) + [roll, pitch, yaw]
            + list(v) + list(w) + list(cur) + list(wav) + list(kick)
            + list(u) + [dist_on])
    return dict(zip(RECORD_FIELDS, (float(x) for x in vals)))


class Recorder:
    """Thread-safe CSV recorder. The viser Record/Stop buttons call start()/stop()
    (on viser's thread) while the sim loop calls log() each frame (main thread); a
    lock guards the file handle so the two never race."""

    def __init__(self, out_dir, fieldnames=RECORD_FIELDS, tag="teleop"):
        self.out_dir = out_dir
        self.fieldnames = fieldnames
        self.tag = tag
        self._lock = threading.Lock()
        self._f = None
        self._w = None
        self.path = None
        self.n = 0
        self.active = False

    def start(self):
        """Open a new timestamped CSV under a per-day subfolder (out_dir/YYYYMMDD/)
        and begin recording. Idempotent (returns the
        current path if already recording). Atomic: on an I/O failure nothing is left
        half-open (the partial handle is closed) and the error is re-raised, so the
        recorder stays clean and a later start() can retry."""
        with self._lock:
            if self.active:
                return self.path
            ts = time.strftime("%Y%m%d_%H%M%S")
            day = ts.split("_")[0]                          # YYYYMMDD (same date as the file)
            day_dir = os.path.join(self.out_dir, day)       # group recordings by day
            os.makedirs(day_dir, exist_ok=True)
            path = os.path.join(day_dir, f"{ts}_{self.tag}.csv")   # date first: <date>_<traj>_<model>.csv
            f = open(path, "w", newline="")
            try:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()
            except Exception:
                f.close()
                raise
            self.path, self._f, self._w = path, f, w
            self.n = 0
            self.active = True
            return self.path

    def log(self, row):
        """Write one row (a dict over RECORD_FIELDS). No-op if not recording. Flushes
        every 200 rows so an abnormal kill (SIGKILL / native crash, which skips the
        finally: stop()) loses at most that many buffered rows."""
        with self._lock:
            if not self.active:
                return
            self._w.writerow(row)
            self.n += 1
            if self.n % 200 == 0:
                self._f.flush()

    def stop(self):
        """Flush + close the CSV. Returns the saved path, or None if not recording."""
        with self._lock:
            if not self.active:
                return None
            try:
                self._f.flush()
                self._f.close()
            finally:
                self._f = self._w = None
                self.active = False
            return self.path
