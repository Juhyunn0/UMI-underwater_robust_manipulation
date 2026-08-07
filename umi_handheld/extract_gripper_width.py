"""extract_gripper_width.py — jaw opening per frame, from the ArUco tags on the fingers.

Consumes the session's rectified-left images, which is the PRE-WARP grid. That ordering
matters: solvePnP needs the camera model the pixels were actually formed under, and the
warp deliberately changes it. Running detection after the warp would silently apply the
target camera's geometry to source-camera pixels.

Output convention, matched to what UMI_underwater's pipeline consumes
(zarr_adapter.py:201 and batch_demos_with_tracking.py:813-815):

    (T, 1) float32 in [0, 1],  0.0 = fully CLOSED,  1.0 = fully OPEN

Two measurement methods, because the mounting decides which one is even valid
--------------------------------------------------------------------------
`gripper.method: relative_axis` is UMI_underwater's quantity — the MOVING tag's centre
expressed in the REFERENCE tag's frame, one axis of it, sign preserved:

    rel = R_ref^T (t_moving - t_reference)        # width_append.py:234-237
    signal = rel[axis]                            # width_append.py:612-616

`gripper.method: euclidean` is the older ||t_a - t_b|| in the camera frame. It is always
positive, so on a rig where one tag travels PAST the other the distance folds into a V
and two different jaw openings normalise to the same number. It is correct only for a
tag-per-jaw rig whose tags separate monotonically. UMI_underwater's shipped range runs
-3.5 -> +15.5 mm (width_append.py:378-379), i.e. straight through zero, which is exactly
the case euclidean cannot represent.

relative_axis also cancels wrist rotation, because the reference tag — not the camera —
defines the frame the displacement is read in.

Which physical direction "open" is depends on where the tags are mounted, so it is a
config value (`gripper.distance_increases_with_opening`), not an assumption baked into
the code. iPhUMI's rig and UMI_underwater's rig disagree about it.

Every dimension involved here is a placeholder until the real gripper is measured; that
fact travels with the output as `dimensions_are_placeholders`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umi_handheld.camera_model import REPO, CameraModel        # noqa: E402
from umi_handheld.record import load_pipeline_config           # noqa: E402

REFINE = {"subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
          "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
          "none": cv2.aruco.CORNER_REFINE_NONE}


def make_detector(acfg: dict) -> cv2.aruco.ArucoDetector:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, acfg["dictionary"]))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = REFINE[acfg.get("corner_refinement", "subpix")]
    return cv2.aruco.ArucoDetector(d, p)


def marker_object_points(length_m: float) -> np.ndarray:
    """Marker-centred, in the corner order detectMarkers returns: TL, TR, BR, BL."""
    h = length_m / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def tag_pose(corners: np.ndarray, length_m: float, K: np.ndarray,
             dist: np.ndarray):
    """solvePnP for one marker. Returns (rvec, tvec), each (3,), or None.

    The rotation is needed as well as the translation: `relative_axis` reads the
    displacement in the REFERENCE tag's frame, which only exists once that tag's
    orientation is known.
    """
    ok, rvec, tvec = cv2.solvePnP(marker_object_points(length_m),
                                  corners.reshape(-1, 2).astype(np.float64),
                                  K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec.ravel(), tvec.ravel()) if ok else None


AXES = {"x": 0, "y": 1, "z": 2}


def axis_quality(v: np.ndarray, window: int = 9) -> dict:
    """Is this series a stroke, or is it jitter?

    Two obvious metrics both fail, and both failed on real takes here:

      * Peak-to-peak. On a small tag the NOISIEST axis routinely spans the most,
        because solvePnP's out-of-plane rotation is its weakest estimate and the
        reference tag's rotation error lands in `rel` scaled by the tag separation.
        On sessions/grippercalibration_0001 frames 280-660 the true stroke axis y
        spanned 26.7 mm raw while the pure-noise axis z spanned 51.8 mm.

      * Monotonicity (span / total variation). Confounded by how many open/close
        cycles the take holds: N clean cycles cap it at 1/(2N), so a 5-cycle
        calibration take cannot score above ~0.1 even when it is perfect.

    What does separate them is signal-to-noise against the series' own smoothed
    version — a stroke is a large excursion that survives smoothing, jitter is not.
    Checked on both takes: sessions/session_syn (true axis x) scores 21.2 against 1.4
    and 2.4 for the other two; sessions/grippercalibration_0001 frames 280-660 (true
    axis y, confirmed visually against the jaws at frames 539 closed / 592 open)
    scores 14.2 against 9.3 and 2.8. Raw span picks the wrong axis in both.
    """
    x = v[~np.isnan(v)]
    if x.size < 2:
        return {"span_m": 0.0, "span_smoothed_m": 0.0, "noise_rms_m": 0.0,
                "snr": 0.0, "score": 0.0, "direction_changes": 0}
    sm = moving_average(v, max(int(window), 3))
    both = ~np.isnan(v) & ~np.isnan(sm)
    span_s = float(np.nanmax(sm) - np.nanmin(sm)) if np.isfinite(sm).any() else 0.0
    noise = float(np.sqrt(np.mean((v[both] - sm[both]) ** 2))) if both.any() else 0.0
    snr = span_s / noise if noise > 0 else 0.0
    s = np.sign(np.diff(x))
    s = s[s != 0]
    return {"span_m": float(x.max() - x.min()), "span_smoothed_m": span_s,
            "noise_rms_m": noise, "snr": snr, "score": snr,
            "direction_changes": int(np.count_nonzero(np.diff(s) != 0))}


def relative_position(moving, reference) -> np.ndarray:
    """Moving tag's centre in the REFERENCE tag's frame: R_ref^T (t_mov - t_ref).

    The same transform as UMI_underwater's tag0_in_tag1_frame
    (width_append.py:234-237); metres here, millimetres there.
    """
    R_ref, _ = cv2.Rodrigues(reference[0])
    return R_ref.T @ (moving[1] - reference[1])


def interpolate_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear fill of NaN runs no longer than max_gap. Longer runs stay NaN."""
    x = x.copy()
    n = len(x)
    good = np.flatnonzero(~np.isnan(x))
    if good.size < 2:
        return x
    for a, b in zip(good[:-1], good[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            x[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [x[a], x[b]])
    return x


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    out = np.full_like(x, np.nan)
    half = window // 2
    for i in range(len(x)):
        seg = x[max(0, i - half):min(len(x), i + half + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--session", type=Path, default=REPO / "sessions/session_0001")
    ap.add_argument("--out", type=Path, default=None, help="default: <session>/gripper_width.json")
    ap.add_argument("--ground-truth", type=Path, default=None,
                    help="synthetic ground_truth.json, for a reported comparison only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_pipeline_config(a.config)
    gcfg = cfg["gripper"]
    acfg = gcfg["aruco"]
    cam = CameraModel.load(cfg["cameras"]["source"])
    session = json.loads((a.session / "session.json").read_text())
    rows = list(csv.DictReader((a.session / "frames.csv").open()))

    ids_wanted = (int(acfg["left_tag_id"]), int(acfg["right_tag_id"]))
    method = str(gcfg.get("method", "euclidean")).lower()
    if method not in ("relative_axis", "euclidean"):
        print(f"ERROR: gripper.method must be relative_axis or euclidean, got "
              f"{method!r}", file=sys.stderr)
        return 2
    axis_cfg = str(gcfg.get("axis", "auto")).lower()
    if method == "relative_axis" and axis_cfg not in ("auto",) + tuple(AXES):
        print(f"ERROR: gripper.axis must be auto|x|y|z, got {axis_cfg!r}", file=sys.stderr)
        return 2
    ref_id = int(gcfg.get("reference_tag_id", ids_wanted[1]))
    if method == "relative_axis" and ref_id not in ids_wanted:
        print(f"ERROR: gripper.reference_tag_id {ref_id} is not one of the aruco ids "
              f"{ids_wanted}", file=sys.stderr)
        return 2
    mov_id = ids_wanted[0] if ref_id == ids_wanted[1] else ids_wanted[1]

    print(f"session      : {a.session}  ({len(rows)} frames, kind={session['kind']})")
    print(f"camera model : {cam.rel}  verified={cam.verified}")
    print(f"aruco        : {acfg['dictionary']}, {acfg['marker_length_m']*1000:.0f} mm, "
          f"ids L={acfg['left_tag_id']} R={acfg['right_tag_id']}")
    if method == "relative_axis":
        print(f"method       : relative_axis  moving={mov_id} in reference={ref_id} "
              f"frame, axis={axis_cfg}")
    else:
        print("method       : euclidean  ||t_a - t_b|| in the camera frame "
              "(monotone only if the tags never pass their closest approach)")
    # The tag says whether these came off a rig or out of thin air; it used to be
    # hardcoded "[PLACEHOLDER]", which kept printing after the values were measured.
    print(f"z gate       : {gcfg['nominal_tag_z_m']:.3f} +- {gcfg['z_tolerance_m']:.3f} m"
          + ("  [PLACEHOLDER]"
             if gcfg.get("dimensions_are_placeholders", True) else "  [measured]"))
    if a.dry_run:
        print("\n--dry-run: config parsed, nothing read, nothing written.")
        return 0

    first = cv2.imread(str(a.session / rows[0]["left_file"]), cv2.IMREAD_GRAYSCALE)
    H, W = first.shape
    K = cam.K(W, H)
    # On a real rectifiedLeft stream the distortion is already removed and this would be
    # zeros with K = P1. Here the synthetic left is the raw distorted grid, so the
    # model's own distortion is the correct thing to pass.
    # TODO(quality): swap for P1/zeros once a device session exists.
    dist = cam.dist

    det = make_detector(acfg)
    T = len(rows)
    euclid = np.full(T, np.nan)                  # ||t_a - t_b||, camera frame
    signed_x = np.full(T, np.nan)                # iPhUMI's quantity, for reference
    rel = np.full((T, 3), np.nan)                # moving tag in the reference frame
    seen = np.zeros((T, 2), bool)
    # Pre-gate, so the gate itself can be set from data: a tag rejected for being at the
    # wrong Z still reports the Z it was at, which is the number the gate needs.
    tag_z = np.full((T, 2), np.nan)
    n_gated = 0

    for i, row in enumerate(rows):
        img = cv2.imread(str(a.session / row["left_file"]), cv2.IMREAD_GRAYSCALE)
        corners, ids, _ = det.detectMarkers(img)
        if ids is None:
            continue
        found = {}
        for c, mid in zip(corners, ids.ravel().tolist()):
            if mid not in ids_wanted:
                continue
            pose = tag_pose(c, float(acfg["marker_length_m"]), K, dist)
            if pose is None:
                continue
            tag_z[i, ids_wanted.index(mid)] = float(pose[1][2])
            # Outlier gate on the tag-plane distance, after iPhUMI's z_tolerance.
            if abs(pose[1][2] - gcfg["nominal_tag_z_m"]) > gcfg["z_tolerance_m"]:
                n_gated += 1
                continue
            found[mid] = pose
        for j, mid in enumerate(ids_wanted):
            seen[i, j] = mid in found
        if len(found) == 2:
            ta, tb = found[ids_wanted[0]], found[ids_wanted[1]]
            euclid[i] = float(np.linalg.norm(tb[1] - ta[1]))
            signed_x[i] = float(tb[1][0] - ta[1][0])
            rel[i] = relative_position(found[mov_id], found[ref_id])

    # ---- which scalar is the signal ------------------------------------------------
    smoothing = int(gcfg["smoothing_window"])
    quality = {k: axis_quality(rel[:, j], smoothing) for k, j in AXES.items()}
    spans = {k: q["span_m"] for k, q in quality.items()}
    axis = axis_source = None
    if method == "relative_axis":
        best = max(quality, key=lambda k: quality[k]["score"])
        print("axis scores  : " + "  ".join(
            f"{k} snr={q['snr']:.1f} (span {q['span_smoothed_m']*1000:.1f}mm / noise "
            f"{q['noise_rms_m']*1000:.1f}mm)" for k, q in quality.items()))
        if axis_cfg == "auto":
            axis, axis_source = best, "auto (smoothed span / residual noise)"
            print(f"axis         : auto picked '{axis}' — PIN IT in the config once "
                  f"calibrated, an auto pick can differ between takes")
        else:
            axis, axis_source = axis_cfg, "config"
            if quality[best]["snr"] > 0 and best != axis:
                print(f"WARNING: axis '{axis}' has snr {quality[axis]['snr']:.1f} but "
                      f"'{best}' has {quality[best]['snr']:.1f} — is the configured "
                      f"axis the stroke direction?", file=sys.stderr)
        # Below about 5 the axes are indistinguishable from rotation jitter; the two
        # real takes scored 21.2 and 14.2, their noise axes 1.4 to 9.3.
        if 0 < quality[best]["snr"] < 5.0:
            print(f"WARNING: the best axis has snr {quality[best]['snr']:.1f} — that is "
                  f"jitter, not a stroke. Check that both tags are rigidly mounted, "
                  f"that the stroke runs IN the reference tag's plane rather than along "
                  f"its normal, and that the tags are big enough in frame.",
                  file=sys.stderr)
        signal = rel[:, AXES[axis]].copy()
    else:
        signal = euclid.copy()

    raw = signal.copy()
    signal = interpolate_gaps(signal, int(gcfg["max_interp_gap"]))
    n_interp = int(np.count_nonzero(np.isnan(raw) & ~np.isnan(signal)))
    signal = moving_average(signal, int(gcfg["smoothing_window"]))
    width = signal                               # what the normalisation below reads

    cal = gcfg.get("calibration") or {}
    w_min, w_max = cal.get("min_width_m"), cal.get("max_width_m")
    cal_source = "config"
    no_signal = False
    # Under relative_axis these are signed positions on one axis, not widths, so a
    # negative endpoint is legitimate — only max > min is required, which the span
    # check below enforces for both methods.
    finite = width[~np.isnan(width)]
    if w_min is None or w_max is None:
        if finite.size == 0 or float(finite.max() - finite.min()) <= 0:
            # A take with no gripper in shot (a bare-camera recording, say) has no
            # signal to normalise. Aborting here would stop the whole pipeline over a
            # channel the depth-only predictor does not read, so the array is emitted
            # as all-closed and the fact is flagged instead of guessed at.
            no_signal = True
            w_min, w_max = 0.0, 1.0
            cal_source = "NONE — no frame had both tags; output is all-zeros"
        else:
            w_min, w_max = float(finite.min()), float(finite.max())
            cal_source = "this take's observed range (no grippercalibration recording)"

    span = w_max - w_min
    if span <= 0:
        print("ERROR: degenerate calibration range", file=sys.stderr)
        return 2
    # A config range that does not overlap what this take actually saw means the config
    # is describing a different rig, a different axis, or different units. Everything
    # then clips to one end and the output is a plausible-looking constant, which is the
    # exact failure this project's measurement rule exists to catch — so say it.
    if cal_source == "config" and finite.size:
        if finite.max() < w_min or finite.min() > w_max:
            print(f"WARNING: this take's signal spans {finite.min()*1000:.1f} to "
                  f"{finite.max()*1000:.1f} mm but the configured calibration is "
                  f"{w_min*1000:.1f} to {w_max*1000:.1f} mm — no overlap, so every "
                  f"frame clips to one end. Wrong axis, wrong rig, or a stale "
                  f"calibration.", file=sys.stderr)
    if no_signal:
        pct = np.zeros(T)
    else:
        pct = (width - w_min) / span
        if not gcfg.get("distance_increases_with_opening", True):
            pct = 1.0 - pct
    pct = np.clip(pct, 0.0, 1.0)
    # TODO(quality): frames where both tags were never found stay NaN and are filled
    # with 0.0 (= fully closed), which is what UMI_underwater does but is a lie.
    n_nan = int(np.count_nonzero(np.isnan(pct)))
    pct = np.nan_to_num(pct, nan=0.0).astype(np.float32).reshape(T, 1)

    def series(x):
        return [None if np.isnan(v) else float(v) for v in x]

    tag_z_stats = {}
    for j, mid in enumerate(ids_wanted):
        col = tag_z[:, j][~np.isnan(tag_z[:, j])]
        tag_z_stats[str(mid)] = ({"n": int(col.size), "min_m": float(col.min()),
                                  "max_m": float(col.max()), "mean_m": float(col.mean())}
                                 if col.size else {"n": 0})

    out = a.out or (a.session / "gripper_width.json")
    np.save(out.with_suffix(".npy"), pct)
    doc = {
        "frames": T,
        "convention": "0.0 = closed, 1.0 = open",
        "dimensions_are_placeholders": bool(gcfg.get("dimensions_are_placeholders", True)),
        "camera_model": cam.provenance(),
        "grid": "rectified-left (pre-warp)",
        "aruco": acfg,
        "method": {
            "name": method,
            "definition": (
                "R_ref^T (t_moving - t_reference), one axis, sign preserved — "
                "UMI_underwater width_append.py:234-237 and :612-616"
                if method == "relative_axis" else
                "||t_a - t_b|| in the camera frame; always positive, so it is "
                "monotone only while the tags do not pass their closest approach"),
            "reference_tag_id": ref_id if method == "relative_axis" else None,
            "moving_tag_id": mov_id if method == "relative_axis" else None,
            "axis": axis,
            "axis_source": axis_source,
            "component_span_m": spans,
            # span alone cannot separate a stroke from rotation jitter; see axis_quality
            "component_quality": quality,
        },
        "detection": {
            "both_tags": int(np.count_nonzero(seen.all(axis=1))),
            "left_only": int(np.count_nonzero(seen[:, 0] & ~seen[:, 1])),
            "right_only": int(np.count_nonzero(seen[:, 1] & ~seen[:, 0])),
            "neither": int(np.count_nonzero(~seen.any(axis=1))),
            "z_gated": n_gated,
            "interpolated": n_interp,
            "unfilled_set_to_closed": n_nan,
            # PRE-gate, by tag id: the range the z gate should be set to cover. A tag
            # the gate rejected still appears here, which is the whole point.
            "tag_z_m": tag_z_stats,
        },
        "no_gripper_signal": no_signal,
        "calibration": {"min_width_m": w_min, "max_width_m": w_max,
                        "source": cal_source,
                        "distance_increases_with_opening":
                            bool(gcfg.get("distance_increases_with_opening", True))},
        "smoothing_window": gcfg["smoothing_window"],
        "series_note": ("signal_m is the processed signal the pct came from (gaps "
                        "filled, then smoothed). The other series are RAW per-frame "
                        "values, unsmoothed."),
        "signal_m": series(width),
        "relative_position_m": {"x": series(rel[:, 0]), "y": series(rel[:, 1]),
                                "z": series(rel[:, 2])},
        "tag_centre_distance_m": series(euclid),
        "signed_x_difference_m": series(signed_x),
        "normalised_pct": pct.ravel().astype(float).tolist(),
        "npy": str(out.with_suffix(".npy").name),
    }

    if a.ground_truth and a.ground_truth.exists():
        gt = json.loads(a.ground_truth.read_text())
        gt_d = np.array([f["scene"]["gripper"]["tag_centre_distance_m"]
                         for f in gt["frames"]])[:T]
        # Always the euclidean tag-centre distance, whatever `method` is: that is the
        # quantity the synthetic sidecar records, and it is compared RAW so the number
        # measures solvePnP and not the smoother.
        m = ~np.isnan(euclid)
        if not m.any():
            # Nothing was detected, so there is nothing to compare. Reporting this as a
            # check that passed would be worse than reporting no check at all — and the
            # usual cause is a capture rendered with a different dictionary or tag size
            # from the one this config now asks for.
            doc["ground_truth_check"] = {
                "note": "NOT RUN — no frame yielded both tags",
                "frames_compared": 0,
            }
            print("GT check     : NOT RUN — no frame yielded both tags (does the "
                  "capture's tag spec match gripper.aruco?)", file=sys.stderr)
        else:
            err = np.abs(euclid[m] - gt_d[m])
            doc["ground_truth_check"] = {
                "note": "synthetic only; compares RAW recovered tag-centre distance to GT",
                "frames_compared": int(m.sum()),
                "mean_abs_error_mm": float(err.mean() * 1000),
                "max_abs_error_mm": float(err.max() * 1000),
            }
            print(f"GT check     : mean {err.mean()*1000:.2f} mm, max "
                  f"{err.max()*1000:.2f} mm over {int(m.sum())} frames")

    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    d = doc["detection"]
    print(f"detection    : both {d['both_tags']}/{T}, z-gated {d['z_gated']}, "
          f"interpolated {d['interpolated']}, filled-as-closed {d['unfilled_set_to_closed']}")
    for mid, s in tag_z_stats.items():
        print(f"tag {mid} z      : " + (f"{s['min_m']*1000:.0f}-{s['max_m']*1000:.0f} mm "
                                        f"(mean {s['mean_m']*1000:.0f}, n={s['n']}, "
                                        f"pre-gate)" if s["n"] else "never posed"))
    print(f"calibration  : {w_min*1000:.1f} to {w_max*1000:.1f} mm  ({cal_source})")
    print(f"pct          : min {pct.min():.3f} max {pct.max():.3f} -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
