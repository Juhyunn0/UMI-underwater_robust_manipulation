#!/usr/bin/env python3
"""nav_fusion.py — a SECOND camera as a fallback localizer, without a second
world.

The station localizes from the C3 (factory underwater intrinsics, measured
extrinsic). The ROV's own RGB camera also sees the floor mat, but its whole
calibration is [예측] — a tape-measure position, a vendor-FOV focal length and
an open-loop tilt mount — so a pose solved through it lands somewhere near the
C3's answer, not on it. Feeding both to the controller as they come would make
the state estimate jump by that difference every time the source changed, and
the MPC would fly the jump.

Two pieces make the RGB usable anyway:

``BodyAlign`` — the extrinsic error is a CONSTANT body-frame transform, and it
can be LEARNED while both cameras have a fix. With T_mb the body pose in the
map, the RGB solve gives ``T_mb_rgb = T_mc · T_cb_assumed`` where the true
answer is ``T_mb = T_mc · T_cb_true``, so

    T_mb_rgb = T_mb · E,   E = T_bc_true · T_cb_assumed      (constant)

for ANY vehicle pose, as long as the mount does not move. E is averaged from
time-paired (C3, RGB) fixes and divided back out: ``T_mb = T_mb_rgb · E⁻¹``.
Any fixed left-multiplication — the map->NED remap, a first-fix datum shared
by both solvers — cancels in E, which is why it can be built from the NavFix
values rather than from solver internals. And because ``T_bc_true = E ·
T_bc_assumed``, E also MEASURES the second camera's real extrinsic, tilt
included — which is the only angle feedback this mount has.

``FallbackArbiter`` — the policy: the C3 fix always wins; an RGB fix is
forwarded only after the C3 has been silent for ``after_s``, and — unless the
operator opts out — only once E is learned from ``min_pairs`` pairs. Before
that, an RGB fix is held back rather than sent raw: a gap in the state
estimate is what the assembler and the mission already know how to bridge; a
15 cm step is not.

No Qt, no cv2 at import: pure numpy, so the tests can run it bare.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


# =============================================================================
# SE(3) helpers — small, explicit, and exactly the conventions tagnav.py uses
# =============================================================================
def T_of(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def T_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rotvec_of(R: np.ndarray) -> np.ndarray:
    """SO(3) log map (axis * angle). Stable for the small angles E carries."""
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-12:
        return np.zeros(3)
    if ang > math.pi - 1e-6:
        # Near pi the axis is read off the symmetric part; E never gets here
        # in practice (a camera cannot be mounted 180 deg wrong AND still see
        # the same floor), but the function must not return garbage if it does.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        i = int(np.argmax(axis))
        axis = A[:, i] / max(axis[i], 1e-12)
        axis /= max(np.linalg.norm(axis), 1e-12)
        return axis * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (ang / (2.0 * math.sin(ang)))


def R_of_rotvec(w) -> np.ndarray:
    """SO(3) exp map (Rodrigues)."""
    w = np.asarray(w, float).reshape(3)
    ang = float(np.linalg.norm(w))
    if ang < 1e-12:
        return np.eye(3)
    k = w / ang
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def tilt_down_deg(R_frd_cam) -> float:
    """The camera's optical axis, as degrees DOWN from body-forward (FRD).

    Same reading as geometry._tilt_flu writes: a mount tilted 30 deg down has
    z_cv = (cos 30, 0, +sin 30) in FRD (+z down), so atan2(z, x) = +30.
    """
    z = np.asarray(R_frd_cam, float)[:, 2]
    return math.degrees(math.atan2(float(z[2]), float(z[0])))


# =============================================================================
# BodyAlign
# =============================================================================
class BodyAlign:
    """E = T_nb_main⁻¹ · T_nb_second, averaged over time-paired fixes.

    ``note_main`` / ``note_second`` take the NED body poses straight off the
    two NavFix streams. A second-source fix is paired with the main fix whose
    capture time is nearest within ``pair_tol_s``; each pair contributes one
    sample of E, and the estimate is the mean of the last ``window`` samples
    (rotation as the mean rotation vector — E's rotation is the mount error,
    tens of degrees at most, where that mean is well-behaved).

    ``ready`` once ``min_pairs`` samples exist. ``reset`` is the caller's job
    whenever the mount MOVES: E is only constant while the extrinsic is.
    """

    def __init__(self, pair_tol_s: float = 0.08, min_pairs: int = 10,
                 window: int = 60):
        self.pair_tol_s = float(pair_tol_s)
        self.min_pairs = max(1, int(min_pairs))
        self._main: deque = deque(maxlen=128)          # (t, T_nb)
        self._rot: deque = deque(maxlen=int(window))   # rotvec samples of E
        self._trn: deque = deque(maxlen=int(window))   # translation samples
        self.n_total = 0                               # pairs ever, this epoch
        self.reset_reason = "not learned yet"
        self.t_last_pair: float | None = None

    # ------------------------------------------------------------- feeding
    def note_main(self, t: float, T_nb: np.ndarray) -> None:
        self._main.append((float(t), np.asarray(T_nb, float)))

    def note_second(self, t: float, T_nb_second: np.ndarray) -> bool:
        """Pair with the nearest main fix. -> True if a sample was added."""
        if not self._main:
            return False
        t = float(t)
        tm, Tm = min(self._main, key=lambda m: abs(m[0] - t))
        if abs(tm - t) > self.pair_tol_s:
            return False
        E = T_inv(Tm) @ np.asarray(T_nb_second, float)
        self._rot.append(rotvec_of(E[:3, :3]))
        self._trn.append(E[:3, 3].copy())
        self.n_total += 1
        self.t_last_pair = t
        return True

    def reset(self, reason: str) -> None:
        self._rot.clear()
        self._trn.clear()
        self.n_total = 0
        self.reset_reason = str(reason)
        self.t_last_pair = None

    # ------------------------------------------------------------- reading
    @property
    def n_pairs(self) -> int:
        return len(self._trn)

    @property
    def ready(self) -> bool:
        return self.n_pairs >= self.min_pairs

    def E(self) -> np.ndarray | None:
        if not self._trn:
            return None
        return T_of(R_of_rotvec(np.mean(np.asarray(self._rot), axis=0)),
                    np.mean(np.asarray(self._trn), axis=0))

    def spread_mm(self) -> float | None:
        """How well the samples agree: RMS distance of the translation
        samples from their mean, in mm. Vehicle motion inside pair_tol_s and
        the RGB's own noise both land here."""
        if len(self._trn) < 2:
            return None
        P = np.asarray(self._trn)
        return 1e3 * float(np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, 1))))

    def offset_mm(self) -> float | None:
        """|E| translation — how far the raw RGB pose sits from the C3's."""
        E = self.E()
        return None if E is None else 1e3 * float(np.linalg.norm(E[:3, 3]))

    def correct(self, T_nb_second: np.ndarray) -> np.ndarray:
        """T_nb = T_nb_second · E⁻¹ (identity until anything is learned)."""
        E = self.E()
        return (np.asarray(T_nb_second, float) if E is None
                else np.asarray(T_nb_second, float) @ T_inv(E))

    def measured_extrinsic(self, R_bc_assumed, t_bc_assumed):
        """T_bc_true = E · T_bc_assumed -> (R_frd_cam, t_frd_cam), or None."""
        E = self.E()
        if E is None:
            return None
        T = E @ T_of(R_bc_assumed, t_bc_assumed)
        return T[:3, :3].copy(), T[:3, 3].copy()

    def measured_tilt_deg(self, R_bc_assumed, t_bc_assumed) -> float | None:
        ext = self.measured_extrinsic(R_bc_assumed, t_bc_assumed)
        return None if ext is None else tilt_down_deg(ext[0])


# =============================================================================
# FallbackArbiter
# =============================================================================
class FallbackArbiter:
    """When does a second-source fix get FORWARDED as the vehicle's state?

    * never while the main source is live (its last fix younger than
      ``after_s``): the state would otherwise alternate between two solvers;
    * otherwise only once the alignment is ready, unless ``allow_unaligned``.

    ``decide`` returns "forward" or a "hold: ..." reason for the note field.
    """

    def __init__(self, after_s: float = 0.2, allow_unaligned: bool = False):
        self.after_s = float(after_s)
        self.allow_unaligned = bool(allow_unaligned)
        self.t_main: float | None = None
        self.t_forwarded: float | None = None     # last second-source forward
        self.n_forwarded = 0

    def note_main_fix(self, t_now: float) -> None:
        self.t_main = float(t_now)

    def main_silent(self, t_now: float) -> bool:
        return self.t_main is None or (float(t_now) - self.t_main) > self.after_s

    def covering(self, t_now: float) -> bool:
        """Is the second source currently standing in for a silent main?"""
        return (self.t_forwarded is not None
                and (float(t_now) - self.t_forwarded) <= self.after_s
                and self.main_silent(t_now))

    def decide(self, t_now: float, aligned: bool) -> str:
        if not self.main_silent(t_now):
            return "hold: C3 live"
        if not aligned and not self.allow_unaligned:
            return "hold: not aligned yet"
        self.t_forwarded = float(t_now)
        self.n_forwarded += 1
        return "forward"
