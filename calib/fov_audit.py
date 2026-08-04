#!/usr/bin/env python3
"""fov_audit.py — what field of view does a stored camera calibration actually imply?

Why this exists
---------------
On 2026-08-04 the question "is the MarineSitu C3's on-EEPROM calibration an in-air or an
underwater calibration?" was answered from a hardcoded comment in our own source
(`c3_collect.py:330` said "IN-AIR"), which cited nothing. That was wrong. This script
replaces the comment with a measurement.

Method
------
A stored calibration is K plus a distortion vector. Neither one states a field of view.
But the pair defines, for every ray direction, where that ray lands on the sensor. So the
FOV is recoverable: sweep the ray angle theta from the optical axis, forward-project with
`cv2.projectPoints`, and find the theta whose image lands exactly on the image border.

Forward projection is used deliberately. `cv2.undistortPoints` inverts the distortion
iteratively and can quietly fail to converge for strongly distorted wide lenses, which is
exactly the regime under test. The forward model is a closed-form polynomial evaluation
and has no such failure mode. The sweep also checks that the projection stays monotonic
over the swept range; a non-monotonic model has left its valid domain and the result is
reported as unreliable rather than returned.

    r_u = tan(theta)                                     ideal (undistorted) radius
    r_d = r_u * (1 + k1 s + k2 s^2 + k3 s^3)             OpenCV rational model,
                / (1 + k4 s + k5 s^2 + k6 s^3),  s = r_u^2
    u   = cx + fx * r_d                                  (horizontal case)

Flat-port refraction
--------------------
A flat viewport obeys Snell's law at the window, so an in-air ray at angle theta_a enters
the housing as theta_w with

    sin(theta_a) = n * sin(theta_w),     n ~ 1.333 for seawater/freshwater

A camera calibrated underwater therefore reports a NARROWER field of view than the same
lens in air. That asymmetry is what distinguishes the two calibrations, and it is large:
for this lens the two hypotheses are ~41 degrees apart, far outside any plausible
calibration-quality error.

Self-tests
----------
The conclusion in FOV_AUDIT.md rests entirely on this script, so `--selftest` runs four
controls, including one that proves the script is capable of returning ~127 deg (i.e. that
the ~85 deg reading is not a structural artifact of the code or of the rational model).

Usage
-----
    python calib/fov_audit.py --audit                  # table over the C3 dumps
    python calib/fov_audit.py --selftest               # controls
    python calib/fov_audit.py --calib <path.json>      # audit one dump
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

N_WATER = 1.333

# In-air vendor specs for the Luxonis OAK-D (Pro) W, the camera inside the MarineSitu C3.
# These are SPEC SHEET values, not measurements — they are the hypothesis being tested.
AIR_SPEC_HFOV = {
    "CAM_A": 95.0,    # IMX378 colour, wide
    "CAM_B": 127.0,   # OV9282 mono left
    "CAM_C": 127.0,   # OV9282 mono right
}
CAM_LABEL = {"CAM_A": "colour IMX378", "CAM_B": "mono left OV9282", "CAM_C": "mono right OV9282"}


# --------------------------------------------------------------------------- core

def project_radius(theta_deg, K, dist, axis="h"):
    """Pixel offset from the principal point for a ray at `theta_deg` off-axis."""
    t = np.tan(np.deg2rad(theta_deg))
    d = (1.0, 0.0) if axis == "h" else (0.0, 1.0)
    pt = np.array([[[d[0] * t, d[1] * t, 1.0]]], dtype=np.float64)
    uv, _ = cv2.projectPoints(pt, np.zeros(3), np.zeros(3), K, dist)
    u, v = uv.ravel()
    return float(np.hypot(u - K[0, 2], v - K[1, 2]))


def monotonic_upto(K, dist, axis="h", hi=85.0, step=0.5):
    """Largest angle up to `hi` below which the projection is strictly increasing."""
    prev, last = -np.inf, 0.0
    for th in np.arange(0.0, hi + 1e-9, step):
        r = project_radius(th, K, dist, axis)
        if not np.isfinite(r) or r < prev:
            return last
        prev, last = r, th
    return hi


def half_angle_at_radius(K, dist, target_px, axis="h", hi=None):
    """Bisect for the ray angle whose projected radius equals `target_px`."""
    if hi is None:
        hi = monotonic_upto(K, dist, axis)
    if project_radius(hi, K, dist, axis) < target_px:
        return float("nan")            # border never reached inside the valid domain
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if project_radius(mid, K, dist, axis) < target_px:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def measure_fov(K, dist, width, height):
    """HFOV / VFOV / DFOV in degrees implied by (K, dist) over a width x height image."""
    cx, cy = K[0, 2], K[1, 2]
    # distance from the principal point to the nearest point on each border
    half_w = min(cx, width - 1 - cx)
    half_h = min(cy, height - 1 - cy)
    corner = float(np.hypot(half_w, half_h))
    h = 2.0 * half_angle_at_radius(K, dist, half_w, "h")
    v = 2.0 * half_angle_at_radius(K, dist, half_h, "v")
    d = 2.0 * half_angle_at_radius(K, dist, corner, "h")
    return {
        "hfov": h, "vfov": v, "dfov": d,
        "valid_upto_deg": monotonic_upto(K, dist, "h"),
        "half_w_px": half_w, "half_h_px": half_h,
    }


def snell_air_from_water(theta_w_deg, n=N_WATER):
    """In-air ray angle that refracts to `theta_w_deg` inside a flat port."""
    s = n * np.sin(np.deg2rad(theta_w_deg))
    return float(np.rad2deg(np.arcsin(s))) if abs(s) <= 1.0 else float("nan")


def snell_water_from_air(theta_a_deg, n=N_WATER):
    """In-water ray angle for an in-air ray at `theta_a_deg` through a flat port."""
    return float(np.rad2deg(np.arcsin(np.sin(np.deg2rad(theta_a_deg)) / n)))


# ------------------------------------------------------------------- model fitting

def fit_rational_radial(theta_deg, r_px):
    """Fit fx and OpenCV rational k1..k6 to a radial mapping (theta -> pixel radius).

    fx is the paraxial limit d(r_px)/d(tan theta) at theta -> 0, taken from the smallest
    sampled angles. With fx fixed, the rational model is LINEAR in k1..k6:

        y = r_px / (fx * r_u),  s = r_u^2
        y * (1 + k4 s + k5 s^2 + k6 s^3) = 1 + k1 s + k2 s^2 + k3 s^3
        =>  y - 1 = [s, s^2, s^3, -y s, -y s^2, -y s^3] . [k1..k6]

    so it solves in one `lstsq`, with no scipy dependency.
    """
    theta = np.asarray(theta_deg, float)
    r = np.asarray(r_px, float)
    keep = theta > 1e-9
    theta, r = theta[keep], r[keep]
    ru = np.tan(np.deg2rad(theta))

    near = theta <= max(3.0, np.percentile(theta, 5))
    fx = float(np.sum(r[near] * ru[near]) / np.sum(ru[near] ** 2))

    y = r / (fx * ru)
    s = ru ** 2
    A = np.stack([s, s ** 2, s ** 3, -y * s, -y * s ** 2, -y * s ** 3], axis=1)
    k, *_ = np.linalg.lstsq(A, y - 1.0, rcond=None)
    dist = np.array([[k[0], k[1], 0.0, 0.0, k[2], k[3], k[4], k[5]]])
    return fx, dist


def equidistant_radius(theta_deg, f_px):
    """Ideal fisheye r = f * theta — a realistic wide-lens mapping for synthetic tests."""
    return f_px * np.deg2rad(np.asarray(theta_deg, float))


# ------------------------------------------------------------------------ loading

def load_dump(path):
    """Extract {cam: (K, dist, W, H)} from a c3_camera calibration.json dump."""
    d = json.load(open(path))
    out = {}
    for cam, c in d.get("device", {}).get("intrinsics", {}).items():
        W, H = c["default_size"]
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)
        out[cam] = (K, np.array(c["distortion"], float).reshape(1, -1), W, H)
    return out, d.get("source", ""), d.get("note", "")


# -------------------------------------------------------------------------- audit

def audit(paths, n=N_WATER):
    rows, sigs = [], {}
    for p in paths:
        cams, src, _ = load_dump(p)
        sig = json.dumps({k: [v[0].tolist(), v[1].tolist()] for k, v in cams.items()}, sort_keys=True)
        sigs.setdefault(sig, []).append(p)
    if len(sigs) > 1:
        print(f"WARNING: {len(sigs)} DIFFERENT calibrations across {len(paths)} dumps", file=sys.stderr)
    rep = sorted(next(iter(sigs.values())))[-1]
    cams, src, _ = load_dump(rep)
    print(f"representative dump : {os.path.relpath(rep)}")
    print(f"identical dumps     : {len(sigs[next(iter(sigs))])} / {len(paths)}")
    print(f"source              : {src}\n")

    hdr = f"{'camera':<24} {'size':>11} {'fx':>9} {'measured':>9} {'air spec':>9} {'water pred':>11} {'delta':>7} {'valid<=':>8}"
    print(hdr); print("-" * len(hdr))
    for cam in ("CAM_A", "CAM_B", "CAM_C"):
        if cam not in cams:
            continue
        K, dist, W, H = cams[cam]
        m = measure_fov(K, dist, W, H)
        air = AIR_SPEC_HFOV[cam]
        pred = 2.0 * snell_water_from_air(air / 2.0, n)
        rows.append(dict(cam=cam, label=CAM_LABEL[cam], W=W, H=H, fx=K[0, 0],
                         hfov=m["hfov"], vfov=m["vfov"], dfov=m["dfov"],
                         air_spec=air, water_pred=pred, delta=m["hfov"] - pred,
                         valid=m["valid_upto_deg"],
                         air_implied=2.0 * snell_air_from_water(m["hfov"] / 2.0, n)))
        print(f"{cam + ' (' + CAM_LABEL[cam] + ')':<24} {str(W) + 'x' + str(H):>11} "
              f"{K[0,0]:9.1f} {m['hfov']:8.1f}d {air:8.1f}d {pred:10.1f}d "
              f"{m['hfov']-pred:+6.1f}d {m['valid_upto_deg']:7.1f}d")

    print("\nreverse check — measured in-water FOV converted back to air via Snell:")
    for r in rows:
        print(f"  {r['cam']}: {r['hfov']:.1f}d in water  ->  {r['air_implied']:.1f}d in air   "
              f"(spec {r['air_spec']:.0f}d, delta {r['air_implied']-r['air_spec']:+.1f}d)")

    worst = max(abs(r["delta"]) for r in rows)
    worst_air = max(abs(r["hfov"] - r["air_spec"]) for r in rows)
    print(f"\nworst |measured - water prediction| = {worst:.1f} deg")
    print(f"worst |measured - air spec|         = {worst_air:.1f} deg")
    print("VERDICT: " + ("UNDERWATER calibration" if worst < worst_air / 3 else "INCONCLUSIVE"))
    return rows


# ----------------------------------------------------------------------- controls

def selftest(verbose=True):
    """Four controls. The conclusion rests on this script, so the script gets audited too."""
    ok = True

    def check(name, got, want, tol, extra=""):
        nonlocal ok
        good = np.isfinite(got) and abs(got - want) <= tol
        ok &= good
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got:.2f}d, want {want:.2f}+-{tol}d {extra}")
        return good

    print("C1  pinhole, zero distortion — does the bisection recover an exact FOV?")
    for target in (60.0, 95.0, 127.0):
        W, H = 1280, 800
        fx = (W / 2) / np.tan(np.deg2rad(target / 2))
        K = np.array([[fx, 0, (W - 1) / 2], [0, fx, (H - 1) / 2], [0, 0, 1]], float)
        m = measure_fov(K, np.zeros((1, 8)), W, H)
        # half-width is (W-1)/2 not W/2, so the exact expectation shifts a hair
        want = 2 * np.rad2deg(np.arctan(((W - 1) / 2) / fx))
        check(f"pinhole {target:.0f}d", m["hfov"], want, 0.01)

    print("\nC2  CRITICAL — can a *genuinely 127 deg* wide camera read back as 127 deg?")
    print("    (if this fails, the ~85d reading is a script/model artifact, not a measurement)")
    W, H = 1280, 800
    true_h = 127.0
    half_w = (W - 1) / 2
    f_eq = half_w / np.deg2rad(true_h / 2)                 # equidistant fisheye
    th = np.linspace(0.0, true_h / 2, 400)
    fx, dist = fit_rational_radial(th, equidistant_radius(th, f_eq))
    K = np.array([[fx, 0, half_w], [0, fx, (H - 1) / 2], [0, 0, 1]], float)
    m = measure_fov(K, dist, W, H)
    check("synthetic 127d fisheye", m["hfov"], true_h, 2.0,
          f"(fitted fx={fx:.1f}, valid<={m['valid_upto_deg']:.0f}d)")

    print("\nC3  closed loop on the REAL C3 distortion: apply Snell, refit, remeasure.")
    print("    the same physical lens in air must read ~127d if the stored model is in-water")
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..",
                                          "c3_camera/datasets/*/calibration.json")))
    if not paths:
        print("  [SKIP] no C3 dumps found")
    else:
        cams, _, _ = load_dump(paths[-1])
        for cam in ("CAM_B", "CAM_C"):
            K, dist, W, H = cams[cam]
            lim = monotonic_upto(K, dist, "h")
            th_w = np.linspace(0.0, min(lim, 46.0), 300)
            r = np.array([project_radius(t, K, dist, "h") for t in th_w])
            th_a = np.array([snell_air_from_water(t) for t in th_w])
            good = np.isfinite(th_a)
            fx_a, dist_a = fit_rational_radial(th_a[good], r[good])
            Ka = np.array([[fx_a, 0, K[0, 2]], [0, fx_a, K[1, 2]], [0, 0, 1]], float)
            ma = measure_fov(Ka, dist_a, W, H)
            check(f"{cam} refitted to air", ma["hfov"], AIR_SPEC_HFOV[cam], 4.0,
                  f"(fx {K[0,0]:.0f} -> {fx_a:.0f})")

    print("\nC4  guard — a model driven past its valid domain must be reported, not returned.")
    W, H = 1280, 800
    K = np.array([[300.0, 0, (W - 1) / 2], [0, 300.0, (H - 1) / 2], [0, 0, 1]], float)
    bad = np.array([[-3.0, 5.0, 0, 0, -4.0, 0, 0, 0]])     # deliberately non-monotonic
    lim = monotonic_upto(K, bad, "h")
    good = lim < 85.0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] non-monotonic model flagged: valid only up to {lim:.1f}d")

    print("\n" + ("ALL CONTROLS PASSED" if ok else "SOME CONTROLS FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true", help="audit the C3 calibration dumps")
    ap.add_argument("--calib", type=str, default=None, help="audit one calibration.json")
    ap.add_argument("--selftest", action="store_true", help="run the controls")
    ap.add_argument("--n", type=float, default=N_WATER, help="refractive index of the water")
    a = ap.parse_args()

    if a.selftest:
        return 0 if selftest() else 1
    paths = [a.calib] if a.calib else sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "c3_camera/datasets/*/calibration.json")))
    if not paths:
        print("no calibration dumps found", file=sys.stderr)
        return 2
    audit(paths, a.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
