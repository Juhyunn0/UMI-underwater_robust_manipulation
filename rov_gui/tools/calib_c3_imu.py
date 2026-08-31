#!/usr/bin/env python3
"""calib_c3_imu.py — the two bench calibrations the dead reckoner cannot run without.

    # 1. accel scale + bias, from a tumble in air. HOLD each attitude 4-5 s:
    #    the fit needs gravity seen from many directions, and a vehicle in
    #    MOTION is not showing it gravity, it is showing it your hands.
    python -m rov_gui.tools.calib_c3_imu <run>/..._c3_imu.jsonl --fit accel \\
        -o config/c3_imu_calib.json

    # 2. the IMU -> body rotation, from 60 s of hand wiggle with ArduSub live
    python -m rov_gui.tools.calib_c3_imu <run>/..._c3_imu.jsonl --fit rotation \\
        --rov <run>/..._rov.jsonl -o config/c3_imu_calib.json

Both write into the SAME JSON, merging rather than overwriting, so the two
sessions can happen on different days. ``--fit both`` does them together when
one recording covers both (a tumble is also a wiggle, if it was slow enough).

Why this tool exists at all
---------------------------
Two facts about this camera, both measured and both fatal to an uncalibrated
dead reckoner (KNOWN_ISSUES.md:452-468, c3_camera/imu.py:22-26):

1. the accelerometer carries a large offset — MEASURED by this tool on
   2026-08-17 as 1.80 m/s^2 (0.184 g), almost entirely on IMU x, with the
   scale within 0.9% of unity;
2. ``getImuToCameraExtrinsics()`` raises, so the device does not know how the
   IMU is oriented relative to anything.

(1) uncorrected is 0.5*1.80*t^2 = 90 m of position error at 10 s [유도] — not
a refinement. NOTE the repo previously recorded this as a "+20% scale error"
from a single upright reading of |a| = 11.8; that pose puts gravity nearly
along IMU x, so the bias read as scale. (2) is worse than it sounds — a wrong
mounting rotation does not blur the estimate, it points it in the wrong
direction, which reads as a plausible drift.

The second fit deliberately goes straight to the BODY frame rather than to the
camera. It needs no IMU-to-camera extrinsic (there isn't one), and the camera's
40 degree mount tilt is absorbed into the answer automatically.

Offline only: matplotlib is fine here, as in plot_nav_run.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.control.imu_dr import G, orthonormalize   # noqa: E402

SCHEMA = 1


# =============================================================================
# input
# =============================================================================
def read_c3_jsonl(path: Path) -> dict:
    """The station's ``*_c3_imu.jsonl`` -> arrays. Keeps the DEVICE clock."""
    t, a, g, acc = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("msg") != "IMU":
                continue
            td = r.get("t_device")
            if td is None:
                continue
            row = (r.get("ax"), r.get("ay"), r.get("az"),
                   r.get("gx"), r.get("gy"), r.get("gz"))
            if any(v is None for v in row):
                continue
            t.append(float(td))
            a.append([float(v) for v in row[:3]])
            g.append([float(v) for v in row[3:]])
            acc.append(f"{r.get('accel_accuracy', '')}/"
                       f"{r.get('gyro_accuracy', '')}")
    if not t:
        raise SystemExit(f"no usable IMU records in {path}")
    order = np.argsort(np.asarray(t))
    return {"t": np.asarray(t)[order], "a": np.asarray(a)[order],
            "g": np.asarray(g)[order], "accuracy": sorted(set(acc))}


def read_rov_jsonl(path: Path) -> dict:
    """``*_rov.jsonl`` -> the autopilot's body rates on the HOST clock.

    ATTITUDE is the source, not RAW_IMU/SCALED_IMU2: the station's own state
    assembler takes body rates from ATTITUDE for the same reason (they are
    the autopilot's filtered, de-biased solution rather than a raw report).
    """
    t, w, rp = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("msg_type") != "ATTITUDE" and r.get("msg") != "ATTITUDE":
                continue
            row = (r.get("rollspeed"), r.get("pitchspeed"), r.get("yawspeed"))
            if any(v is None for v in row):
                continue
            t.append(float(r.get("t")))
            w.append([float(v) for v in row])
            rp.append([float(r.get("roll") or 0.0),
                       float(r.get("pitch") or 0.0)])
    if not t:
        raise SystemExit(f"no ATTITUDE records in {path}")
    order = np.argsort(np.asarray(t))
    return {"t": np.asarray(t)[order], "w": np.asarray(w)[order],
            "rp": np.asarray(rp)[order]}


# =============================================================================
# 1. accelerometer: scale + bias by ellipsoid fit
# =============================================================================
def fibonacci_bins(u: np.ndarray, n_bin: int = 64) -> int:
    """How many of ``n_bin`` roughly-equal-area directions the data covers.

    Coverage, not sample count, is what makes this fit well posed: two minutes
    of a vehicle lying still is 24000 samples of one direction and will still
    produce three confident, wrong scale factors.
    """
    i = np.arange(n_bin) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_bin)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    ref = np.stack([np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    hit = (u @ ref.T).argmax(axis=1)
    return int(np.unique(hit).size)


# What a held attitude LOOKS LIKE to an operator, per body half-axis. The
# measured specific force points along whichever body axis is currently UP: at
# rest f = -R^T g, and FRD z is down, so a normal upright vehicle reads -z.
# Named in body/FRD terms and not IMU terms because the person holding the
# vehicle is looking at the vehicle, not at the chip.
POSE_NAME = {
    "+x": "nose up", "-x": "nose down",
    "+y": "lying on its LEFT side", "-y": "lying on its RIGHT side",
    "+z": "upside down", "-z": "normal upright",
}


def axis_coverage(u: np.ndarray, thr: float = 0.5) -> dict:
    """Samples per body half-axis — the observability condition for a
    diagonal scale+bias. An axis that never points up AND down leaves its own
    scale factor undetermined, and the solver will still return a number."""
    out = {}
    for i, ax in enumerate("xyz"):
        out[f"+{ax}"] = int((u[:, i] > thr).sum())
        out[f"-{ax}"] = int((u[:, i] < -thr).sum())
    return out


def static_mask(a: np.ndarray, w: np.ndarray, t: np.ndarray | None = None,
                w_max: float = 0.05, sd_max: float = 0.1,
                window_s: float = 0.5) -> np.ndarray:
    """The samples where the vehicle was actually HELD STILL at an attitude.

    A gyro gate alone is not enough and this is not a subtlety — it is the
    difference between a calibration and a fiction. Rotating a 13.7 kg vehicle
    by hand imparts several m/s^2 of TRANSLATIONAL acceleration that no gyro
    threshold can see: measured on the first real tumble, |a| reached
    27.8 m/s^2 (2.8 g) while |gyro| stayed under the old 0.5 rad/s gate, and
    the fit dutifully absorbed the handling as a 1.8 m/s^2 accelerometer bias
    [측정: sessions/low_level_controller_data/20260817/0817_094728/
    c3_depth_20260817_094733_c3_imu.jsonl].

    So the real test is STEADINESS: over a short window the accelerometer
    magnitude must barely move. Gravity held at a fixed attitude is constant;
    a vehicle being carried between attitudes is not.
    """
    n = np.linalg.norm(a, axis=1)
    hz = (1.0 / max(float(np.median(np.diff(t))), 1e-6)
          if t is not None and t.size > 1 else 200.0)
    k = max(3, int(window_s * hz))
    if n.size <= k:
        return np.linalg.norm(w, axis=1) < w_max
    # Rolling mean and std of |a|, and rolling mean |gyro|, centred.
    def _roll(v):
        c = np.cumsum(np.insert(v, 0, 0.0))
        return (c[k:] - c[:-k]) / k
    m1, m2 = _roll(n), _roll(n * n)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    wm = _roll(np.linalg.norm(w, axis=1))
    keep = np.zeros(n.size, bool)
    keep[: sd.size] = (sd < sd_max) & (wm < w_max)
    return keep


def fit_accel(a, w, t=None, diag: bool = True, w_max: float = 0.05,
              sd_max: float = 0.1, min_bins: int = 20, R_frd_imu=None) -> dict:
    """Solve ``min sum( ||S(a-b)|| - g )^2`` over the STILL parts of a tumble.

    Only genuinely static samples are used — see :func:`static_mask` for why a
    gyro threshold on its own silently ruins this.

    ``a``/``w``/``t`` may each be a LIST of arrays, one per take. Scale and
    bias are properties of the sensor, not of a recording, so a calibration
    may be assembled from several takes — which is what actually happens when
    the first pass misses one attitude and nobody wants to redo the other
    five. The static mask is computed PER TAKE so its rolling window never
    straddles the join, where the apparent acceleration would jump.
    """
    takes = a if isinstance(a, (list, tuple)) else [a]
    ws = w if isinstance(w, (list, tuple)) else [w]
    ts = t if isinstance(t, (list, tuple)) else [t]
    masks = [static_mask(ai, wi, ti, w_max=w_max, sd_max=sd_max)
             for ai, wi, ti in zip(takes, ws, ts)]
    A = np.concatenate([ai[mi] for ai, mi in zip(takes, masks)], axis=0)
    keep = np.concatenate(masks)
    a = np.concatenate(list(takes), axis=0)
    out = {"model": "diag" if diag else "diag+bias", "samples": int(A.shape[0]),
           "rejected_moving": int((~keep).sum()),
           "gate": {"gyro_max_rad_s": w_max, "accel_sd_max_m_s2": sd_max}}
    if A.shape[0] < 500:
        out["error"] = (f"only {A.shape[0]} genuinely still samples "
                        f"(|w| < {w_max} rad/s AND |a| steady to "
                        f"{sd_max} m/s^2). HOLD the vehicle at each attitude "
                        f"for a second or two instead of swinging it through.")
        return out
    n0 = np.linalg.norm(A, axis=1)
    u = A / np.maximum(n0[:, None], 1e-9)
    out["bins"] = fibonacci_bins(u)
    out["residual_rms_before"] = float(np.sqrt(((n0 - G) ** 2).mean()))
    # The gate is AXIS COVERAGE, not sphere coverage. A diagonal scale+bias is
    # six numbers, and the classic six-position method determines them exactly:
    # each axis pointing up and each pointing down. Demanding 20 of 64
    # Fibonacci bins was a proxy for "well spread" that asks far more than the
    # model needs — and on a 13.7 kg vehicle that a person has to HOLD at each
    # attitude, the difference between 6 poses and 15 is the difference
    # between a calibration that happens and one that does not.
    # Coverage is judged in BODY axes when the mounting rotation is known, so
    # a missing pose can be named as something the operator can actually do
    # ("nose up") instead of as a chip axis nobody can point at.
    if R_frd_imu is not None:
        out["axis_coverage"] = axis_coverage(u @ np.asarray(R_frd_imu).T)
        out["coverage_frame"] = "body FRD (R_frd_imu applied)"
    else:
        out["axis_coverage"] = axis_coverage(u)
        out["coverage_frame"] = "IMU axes (no R_frd_imu yet — fit the "\
                                "rotation first for named poses)"
    missing = [k for k, v in out["axis_coverage"].items() if v < 100]
    if missing:
        out["error"] = ("the tumble never held these attitudes: "
                        + ", ".join(f"{k} up ({POSE_NAME[k]})"
                                    for k in missing)
                        + ". Each axis must point UP and DOWN at some point "
                          "or the scale on it is not observable — hold each "
                          "for ~4 s.")
        return out
    if out["bins"] < min_bins:
        out["note"] = (f"only {out['bins']}/64 sphere bins, but all six "
                       f"half-axes are covered, which is what a diagonal "
                       f"scale+bias needs")

    # Linear least squares in the standard ellipsoid parameterisation:
    #   sum_k q_k a_k^2 + l_k a_k + c = 0     (7 unknowns, solved as the null
    #                                          vector of the design matrix)
    # then s and b fall out of q and l.
    #
    # Done in units of g, and that is not cosmetic: in m/s^2 the quadratic
    # columns are ~96 and the constant column is 1, so the condition number
    # runs to 1e16 on perfectly good data and the guard below would reject
    # every real tumble. Normalised, all seven columns are O(1) and the
    # condition number reports the DATA's degeneracy, which is the thing worth
    # refusing on.
    An = A / G
    x, y, z = An[:, 0], An[:, 1], An[:, 2]
    M = np.stack([x * x, y * y, z * z, x, y, z, np.ones_like(x)], axis=1)
    _u, sv, vt = np.linalg.svd(M, full_matrices=False)
    # NOT the condition number. This is a null-space problem: data lying on a
    # good ellipsoid drives the SMALLEST singular value to zero, so a large
    # sv[0]/sv[-1] is the success case and rejecting on it would reject every
    # clean tumble (it did, until this was written down). What has to be
    # checked is that the null space is ONE-dimensional — i.e. the
    # second-smallest value is still well clear of zero. When it is not, more
    # than one ellipsoid fits the data equally well and the solver picks one.
    gap = float(sv[-2] / max(sv[0], 1e-30))
    out["null_gap"] = gap
    out["cond"] = float(sv[0] / max(sv[-1], 1e-30))
    if gap < 1e-6:
        out["error"] = (f"the ellipsoid is not determined (null-space gap "
                        f"{gap:.1e}) — more than one fits these directions")
        return out
    c = vt[-1]                       # M @ c = 0, smallest right singular vector
    q = c[:3]
    if np.any(np.abs(q) < 1e-12):
        out["error"] = "degenerate quadratic term"
        return out
    b = -c[3:6] / (2.0 * q)          # centre, in units of g
    k = float(np.sum(q * b * b) - c[6])
    # Only the RATIO may be sign-checked. A null vector is defined up to sign,
    # so q and k flip together and q/k does not; testing k > 0 on its own
    # rejects half of all correct answers depending on which sign the SVD
    # happened to return.
    if abs(k) < 1e-30 or np.any(q / k <= 0):
        out["error"] = ("fit did not produce a real ellipsoid — the surface "
                        "through these points is not closed")
        return out
    s = np.sqrt(q / k)               # dimensionless, applied to raw m/s^2
    b = b * G                        # ...and the centre back into m/s^2
    corrected = (A - b) * s
    out["residual_rms_after"] = float(np.sqrt(
        ((np.linalg.norm(corrected, axis=1) - G) ** 2).mean()))
    out["scale"] = [float(v) for v in s]
    out["bias"] = [float(v) for v in b]
    out["mean_magnitude_before"] = float(n0.mean())
    return out


# =============================================================================
# 2. IMU -> body rotation, by Kabsch on angular velocity
# =============================================================================
def estimate_lag(t_a, wa, t_b, wb, span_s=1.0, step_s=0.001,
                 hz=100.0) -> tuple[float, float]:
    """Seconds to add when SAMPLING the autopilot stream (``t_b``) so that it
    lines up with the C3 stream (``t_a``).

    Sign, spelled out because it is easy to get backwards and impossible to
    notice afterwards: a POSITIVE result means the autopilot's copy of an
    event carries a later timestamp than the C3's, so the autopilot is read
    later to catch the same motion. ``fit_rotation`` consumes it in exactly
    that form.

    Done FIRST and refused if it is not sharp, because a clock offset and a
    rotation are indistinguishable in the fit that follows: both make the two
    rate vectors disagree, and Kabsch will happily absorb one as the other.
    """
    t0 = max(t_a[0], t_b[0]) + span_s
    t1 = min(t_a[-1], t_b[-1]) - span_s
    if t1 - t0 < 5.0:
        return 0.0, 0.0
    grid = np.arange(t0, t1, 1.0 / hz)
    na = np.interp(grid, t_a, np.linalg.norm(wa, axis=1))
    na = na - na.mean()
    best, best_r = 0.0, -2.0
    for lag in np.arange(-span_s, span_s + 1e-9, step_s):
        nb = np.interp(grid + lag, t_b, np.linalg.norm(wb, axis=1))
        nb = nb - nb.mean()
        d = float(np.linalg.norm(na) * np.linalg.norm(nb))
        if d < 1e-9:
            continue
        r = float(na @ nb) / d
        if r > best_r:
            best, best_r = float(lag), r
    return best, best_r


def fit_rotation(c3: dict, rov: dict, hz: float = 100.0,
                 min_corr: float = 0.8, max_sv_ratio: float = 10.0) -> dict:
    """R_frd_imu from ``w_body ~ R @ w_imu`` over a hand wiggle."""
    lag, corr = estimate_lag(c3["t"], c3["g"], rov["t"], rov["w"])
    out = {"method": "kabsch on angular velocity", "lag_ms": lag * 1000.0,
           "lag_correlation": corr, "rate_hz": hz}
    if corr < min_corr:
        out["error"] = (f"the two rate streams do not line up (peak "
                        f"correlation {corr:.2f} < {min_corr}). Wiggle the "
                        f"vehicle by hand about all three axes with BOTH logs "
                        f"running; a still recording has nothing to align.")
        return out
    t0 = max(c3["t"][0], rov["t"][0] - lag) + 0.5
    t1 = min(c3["t"][-1], rov["t"][-1] - lag) - 0.5
    grid = np.arange(t0, t1, 1.0 / hz)
    out["samples"] = int(grid.size)
    if grid.size < 500:
        out["error"] = f"only {grid.size} overlapping samples"
        return out
    wi = np.stack([np.interp(grid, c3["t"], c3["g"][:, k]) for k in range(3)], 1)
    wb = np.stack([np.interp(grid + lag, rov["t"], rov["w"][:, k])
                   for k in range(3)], 1)
    # Each stream's own mean removed: a constant bias on either side is not a
    # rotation and must not be fitted as one.
    wi = wi - wi.mean(axis=0)
    wb = wb - wb.mean(axis=0)
    H = wi.T @ wb
    U, sv, Vt = np.linalg.svd(H)
    out["singular_values"] = [float(v) for v in sv]
    ratio = float(sv[0] / max(sv[-1], 1e-30))
    if ratio > max_sv_ratio:
        out["error"] = (f"the wiggle was not three-dimensional (singular "
                        f"value ratio {ratio:.1f} > {max_sv_ratio}) — one "
                        f"axis carries almost no signal, so R is not "
                        f"determined. Rotate about all three axes.")
        return out
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    R = orthonormalize(R)
    resid = wb - wi @ R.T
    scale = float(np.sqrt((wb ** 2).sum(axis=1).mean()))
    # NOT the accuracy of R. This is the NOISE FLOOR of comparing the C3's raw
    # gyro against ArduSub's filtered ATTITUDE rates: different filters,
    # different phase response, MAVLink arrival jitter. Measured 2026-08-17 it
    # sits at 6-8 deg and does NOT shrink with a more vigorous wiggle, while
    # two independent takes produced rotations agreeing to 0.81 deg.
    #
    # So do not chase this number. The two things that DO say whether R is
    # right are the gravity cross-check below and, best of all, repeating the
    # take and comparing the two matrices.
    out["rms_deg"] = float(math.degrees(
        np.sqrt((resid ** 2).sum(axis=1).mean()) / max(scale, 1e-9)))
    out["rms_deg_note"] = ("noise floor of raw-gyro vs filtered-ATTITUDE, "
                           "NOT the accuracy of R — repeat the take and "
                           "compare matrices for that")
    out["R_frd_imu"] = [[float(v) for v in row] for row in R]
    # How far the answer is from a plain axis permutation. NOT snapped: if the
    # IMU rides a camera tilted 40 degrees, the true rotation is not one, and
    # snapping would replace a measurement with a guess.
    snap = np.zeros((3, 3))
    for i in range(3):
        j = int(np.argmax(np.abs(R[i])))
        snap[i, j] = math.copysign(1.0, R[i, j])
    if abs(np.linalg.det(snap) - 1.0) < 1e-9:
        ang = math.degrees(math.acos(
            max(-1.0, min(1.0, (np.trace(snap.T @ R) - 1.0) / 2.0))))
        out["nearest_axis_permutation_deg"] = float(ang)
    return out


def gravity_crosscheck(c3: dict, R: np.ndarray, scale, bias,
                       w_still: float = 0.05, rov: dict | None = None,
                       lag: float = 0.0) -> dict:
    """Does the fitted rotation put gravity where the AUTOPILOT says it is?

    Kabsch can return an axis swap or a 180 degree flip from a bad lag, and
    that failure looks like a perfectly good rotation matrix. This is the
    independent check — rotate the corrected accelerometer into the body frame
    and see whether it points down.

    ``rov`` supplies the attitude the vehicle actually held. Without it this
    falls back to assuming the still stretch was LEVEL, which is an assumption
    the caller cannot verify and which was wrong the first time it was used on
    real data: the vehicle sat at its own resting attitude, not level, and the
    check reported a 6.8 degree discrepancy that was the ASSUMPTION being
    wrong rather than the fit.
    """
    still = np.linalg.norm(c3["g"], axis=1) < w_still
    out = {"still_samples": int(still.sum())}
    if still.sum() < 200:
        out["error"] = "no still stretch to check against"
        return out
    f = ((c3["a"][still] * np.asarray(scale)) - np.asarray(bias)) @ np.asarray(R).T
    m = f.mean(axis=0)
    if rov is not None and "rp" in rov:
        # Gravity as the autopilot's own roll/pitch implies it, in body FRD:
        # -R_body_ned^T @ [0,0,g]. No level assumption anywhere.
        ts = c3["t"][still] + lag
        rp = np.stack([np.interp(ts, rov["t"], rov["rp"][:, k])
                       for k in range(2)], axis=1)
        phi, th = float(np.mean(rp[:, 0])), float(np.mean(rp[:, 1]))
        want = np.array([G * math.sin(th),
                         -G * math.sin(phi) * math.cos(th),
                         -G * math.cos(phi) * math.cos(th)])
        out["reference"] = "autopilot ATTITUDE"
        out["still_attitude_deg"] = [math.degrees(phi), math.degrees(th)]
    else:
        want = np.array([0.0, 0.0, -G])
        out["reference"] = "assumed LEVEL (no autopilot attitude given)"
    cosang = float(m @ want / max(np.linalg.norm(m) * np.linalg.norm(want),
                                  1e-9))
    out["mean_body_accel"] = [float(v) for v in m]
    out["expected_body_accel"] = [float(v) for v in want]
    out["angle_from_expected_deg"] = float(
        math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return out


# =============================================================================
# output
# =============================================================================
def merge_write(path: Path, block: dict) -> dict:
    """Merge into an existing calibration rather than replacing it: the accel
    tumble and the rotation wiggle are two different bench sessions and may be
    days apart."""
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except ValueError:
            old = {}
    old.update(block)
    old["schema"] = SCHEMA
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    return old


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path, nargs="+",
                    help="one or more *_c3_imu.jsonl. Several takes are "
                         "merged for --fit accel (scale and bias belong to "
                         "the sensor, not to a recording); --fit rotation "
                         "uses the FIRST one, which must be the take whose "
                         "*_rov.jsonl you pass.")
    ap.add_argument("--fit", choices=("accel", "rotation", "both"),
                    default="accel")
    ap.add_argument("--rov", type=Path, default=None,
                    help="the *_rov.jsonl beside it (required for --fit "
                         "rotation): the autopilot's ATTITUDE rates are the "
                         "reference the IMU is aligned to")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("config/c3_imu_calib.json"))
    ap.add_argument("--cam-tilt-deg", type=float, default=None,
                    help="the camera tilt the vehicle was WEARING for this "
                         "fit. Recorded so a later re-tilt can be seen to "
                         "have invalidated it (the IMU rides the camera).")
    ap.add_argument("--w-max", type=float, default=0.05,
                    help="a still sample's |gyro| ceiling, rad/s "
                         "(default: %(default)s)")
    ap.add_argument("--sd-max", type=float, default=0.1,
                    help="a still sample's rolling |accel| std ceiling, m/s^2 "
                         "(default: %(default)s). THIS is the gate that "
                         "rejects handling: carrying the vehicle between "
                         "attitudes is invisible to a gyro threshold.")
    a = ap.parse_args(argv)

    takes = [read_c3_jsonl(p) for p in a.jsonl]
    c3 = takes[0]
    for p, m in zip(a.jsonl, takes):
        dur = float(m["t"][-1] - m["t"][0])
        print(f"c3   : {m['t'].size} samples, {dur:.1f} s, "
              f"{m['t'].size / max(dur, 1e-9):.0f} Hz, "
              f"accuracy {','.join(m['accuracy'])}  {p.name}")

    block = {"source_c3_jsonl": [str(p) for p in a.jsonl]}
    if a.cam_tilt_deg is not None:
        block["cam_tilt_deg_at_fit"] = float(a.cam_tilt_deg)
    bad = False

    if a.fit in ("accel", "both"):
        prior = {}
        if Path(a.out).exists():
            try:
                prior = json.loads(Path(a.out).read_text())
            except ValueError:
                prior = {}
        acc = fit_accel([m["a"] for m in takes], [m["g"] for m in takes],
                        [m["t"] for m in takes], w_max=a.w_max,
                        sd_max=a.sd_max, R_frd_imu=prior.get("R_frd_imu"))
        acc["takes"] = len(takes)
        block["accel"] = acc
        if "error" in acc:
            print(f"accel: REFUSED — {acc['error']}")
            cov = acc.get("axis_coverage") or {}
            if cov:
                print(f"       held per attitude ({acc.get('coverage_frame')}):")
                for k in ("+x", "-x", "+y", "-y", "+z", "-z"):
                    n = cov.get(k, 0)
                    print(f"         {POSE_NAME[k]:<24} "
                          + ("%5.1f s  ok" % (n / 200.0) if n >= 100
                             else "%5.1f s  <-- NEEDED" % (n / 200.0)))
            bad = True
        else:
            print(f"accel: {acc['samples']} still samples "
                  f"({acc['rejected_moving']} rejected as moving), "
                  f"{acc['bins']}/64 bins, |a| {acc['mean_magnitude_before']:.3f} "
                  f"-> residual {acc['residual_rms_before']:.3f} "
                  f"-> {acc['residual_rms_after']:.3f} m/s^2")
            print(f"       scale {np.round(acc['scale'], 5).tolist()}  "
                  f"bias {np.round(acc['bias'], 4).tolist()}")
            if acc.get("note"):
                print(f"       note: {acc['note']}")

    if a.fit in ("rotation", "both"):
        if a.rov is None:
            ap.error("--fit rotation needs --rov <*_rov.jsonl>")
        rov = read_rov_jsonl(a.rov)
        print(f"rov  : {rov['t'].size} ATTITUDE records, "
              f"{rov['t'][-1] - rov['t'][0]:.1f} s")
        rot = fit_rotation(c3, rov)
        block["R_fit"] = rot
        if "error" in rot:
            print(f"rot  : REFUSED — {rot['error']}")
            bad = True
        else:
            block["R_frd_imu"] = rot["R_frd_imu"]
            print(f"rot  : lag {rot['lag_ms']:+.1f} ms (corr "
                  f"{rot['lag_correlation']:.2f}), residual "
                  f"{rot['rms_deg']:.2f} deg  (noise floor of raw-gyro vs "
                  f"filtered-ATTITUDE, not R's error)")
            if "nearest_axis_permutation_deg" in rot:
                print(f"       {rot['nearest_axis_permutation_deg']:.1f} deg "
                      f"from the nearest axis permutation (NOT snapped)")
            prev = block.get("accel") or {}
            chk = gravity_crosscheck(
                c3, rot["R_frd_imu"],
                prev.get("scale", [1.0] * 3), prev.get("bias", [0.0] * 3),
                rov=rov, lag=rot["lag_ms"] / 1000.0)
            block["gravity_check"] = chk
            ang = chk.get("angle_from_expected_deg")
            if ang is None:
                print(f"       gravity check skipped: {chk.get('error')}")
            else:
                print(f"       gravity check {ang:.1f} deg from level-expected"
                      + ("" if ang < 5.0 else "  <-- SUSPECT (axis swap?)"))
                if ang >= 5.0:
                    bad = True

    out = merge_write(a.out, block)
    print(f"\nwrote {a.out}"
          + ("" if not bad else "  (with the refusal recorded — the dead "
                                "reckoner will run RAW until it is fixed)"))
    if not bad:
        have = ("accel" in out and "scale" in out["accel"],
                "R_frd_imu" in out)
        print(f"  accel scale/bias: {'yes' if have[0] else 'MISSING'}   "
              f"R_frd_imu: {'yes' if have[1] else 'MISSING'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
