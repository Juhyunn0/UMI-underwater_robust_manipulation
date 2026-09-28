#!/usr/bin/env python3
"""
depth_two_cameras.py — the OAK-D-W and the C3, same scene, ONE colour rule.

What this answers
-----------------
"At the same physical distance, do the two cameras land on the same colour?"

That is the check Max asked for on 2026-09-04, reduced to its smallest honest
form. Both cameras look at the same scene at the same moment; both pairs go
through the SAME FoundationStereo session at the SAME settings; both maps are
drawn by the SAME renderer with the SAME fixed endpoints. So any colour
difference left in the picture is a difference in the DEPTH, not in the drawing
— which is exactly what could not be said of the four legacy renderers.

    OAK-D-W  mxid 14442C10716DBCD600, USB   the camera the land demonstrations
                                            were recorded on; calibration IN AIR
    C3       mxid 19443010315B0B2F00, PoE   the deployment camera; its factory
                                            calibration is UNDERWATER

Each camera is rectified by ITS OWN rig — ``umi_dataset.Rectifier(alpha=0.0)``
for the OAK-D-W (the training object) and ``c3_camera.host_depth.StereoRig`` for
the C3 (what the station flies) — and each converts disparity to millimetres
with the focal length of the very images it matched. Nothing is rescaled and no
fudge factor is applied anywhere.

What to expect, and why a mismatch is the POINT
------------------------------------------------
In air these two should NOT agree, and the disagreement is informative rather
than a bug in this tool. The C3's EEPROM calibration is the vendor's UNDERWATER
one (vendor-confirmed; 85.6 deg measured against Snell's 84.3), so used in air
it reads long by roughly the index of refraction — order 1.33x. The OAK-D-W's
calibration is in air and should read true there. Underwater the roles swap.

So: run this in air, expect the C3 to read FARTHER than the OAK-D-W at the same
surface, and read the ratio this tool prints. That ratio is the number the
land-to-water depth argument actually rests on, and until now nothing measured
it on one scene at one moment with one renderer.

``--roi`` restricts the reported statistics to a centre box, which is what you
want when the two cameras have different fields of view and only the middle of
the frame is looking at the same thing.

Usage
-----
    python -m rov_gui.tools.depth_two_cameras --out /tmp/twocam
    python -m rov_gui.tools.depth_two_cameras --out /tmp/twocam --domain metric
    python -m rov_gui.tools.depth_two_cameras --out /tmp/twocam --frames 5 --roi 0.4

This tool opens two cameras and writes files. It never commands the vehicle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rov_gui.tools.depth_compare import (  # noqa: E402
    Z_NEAR_M, Z_FAR_M, CMAPS, colorize, palette_pos, render, side_by_side,
    colour_rule_json, near_row)
from rov_gui.tools.depth_capture_umi import (  # noqa: E402
    LAND_MXID, DEFAULT_DATASET, load_training_rectifier)

#: The deployment camera, on the tether network.
C3_MXID = "19443010315B0B2F00"

#: One FoundationStereo session serves both cameras, at the TRAINING settings.
#: Using one session is not a convenience: it removes the network, the weights,
#: the iteration count and the input scale from the list of things that could
#: explain a difference between the two pictures.
ITERS, SCALE = 16, 1.0

MONO = (640, 400)


def _mono_pipeline(fps: float):
    import depthai as dai
    p = dai.Pipeline()
    for name, sock in (("left", dai.CameraBoardSocket.CAM_B),
                       ("right", dai.CameraBoardSocket.CAM_C)):
        m = p.create(dai.node.MonoCamera)
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setBoardSocket(sock)
        m.setFps(float(fps))
        x = p.create(dai.node.XLinkOut)
        x.setStreamName(name)
        m.out.link(x.input)
    return p


def _open(mxid: str, fps: float, tries: int = 12):
    import depthai as dai
    # Enumeration comes back EMPTY for a second or two after any device on the
    # bus is closed — the PoE unit reboots and the USB one re-enumerates — so a
    # single query is not evidence that a camera is absent. Retry before
    # believing it.
    found = {}
    for k in range(max(1, tries)):
        found = {d.getMxId(): d for d in dai.Device.getAllAvailableDevices()}
        if mxid in found:
            break
        if k + 1 < tries:
            time.sleep(1.5)
    if mxid not in found:
        # getAllAvailableDevices() lists only FREE devices, so a camera another
        # process already holds simply vanishes from it. Say which of the two
        # it is, because "not reachable" sends people to check cables when the
        # actual answer is that their own station is running.
        busy = {}
        try:
            busy = {d.getMxId(): d.state.name
                    for d in dai.XLinkConnection.getAllConnectedDevices()}
        except Exception:                                        # noqa: BLE001
            pass
        if busy.get(mxid) == "X_LINK_BOOTED":
            raise RuntimeError(
                f"{mxid} is present but BOOTED, i.e. some process still owns "
                f"it — DepthAI allows one pipeline per device.\n"
                f"  1. The station is the usual culprit:\n"
                f"       pgrep -af 'python -m rov_gui'\n"
                f"  2. A crashed or SIGKILLed tool leaves the device booted "
                f"and it does NOT always reset itself. Find it by INTERPRETER, "
                f"not by script name — a heredoc run shows up as `python -` "
                f"and no name-based pkill will match it:\n"
                f"       ps aux | grep 'rovgui-pose/bin/python'\n"
                f"     then SIGTERM it (never -9: a clean exit is what returns "
                f"the camera; -9 is how it got stuck).\n"
                f"  3. Only if neither works, replug USB / power-cycle PoE.")
        raise RuntimeError(f"{mxid} not reachable. Free: "
                           + (", ".join(f"{k} ({v.protocol.name})"
                                        for k, v in found.items()) or "none")
                           + "; seen in any state: "
                           + (", ".join(f"{k} ({v})" for k, v in busy.items())
                              or "none"))
    info = found[mxid]
    if info.protocol.name == "X_LINK_TCP_IP":
        return dai.Device(_mono_pipeline(fps), info)
    return dai.Device(_mono_pipeline(fps), info, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)


def _grab(dev, warmup: int):
    """One pair, after letting auto-exposure settle.

    The first frames off these sensors are saturated white, and a saturated pair
    has no texture for the matcher — the depth would be real-looking and
    meaningless.
    """
    ql, qr = dev.getOutputQueue("left", 4, False), dev.getOutputQueue("right", 4, False)
    l = r = None
    for _ in range(max(1, warmup)):
        l, r = ql.get().getCvFrame(), qr.get().getCvFrame()
    return l, r


def _stats(z: np.ndarray, roi: float) -> dict:
    """Percentiles in metres over a centre box of the frame.

    A centre box because the two cameras have different fields of view: the
    edges are looking at different things, and a full-frame median would compare
    one camera's ceiling with the other's wall.
    """
    h, w = z.shape[:2]
    if 0.0 < roi < 1.0:
        dh, dw = int(h * (1 - roi) / 2), int(w * (1 - roi) / 2)
        z = z[dh:h - dh, dw:w - dw]
    v = z[np.isfinite(z) & (z > 0)]
    if v.size == 0:
        return {"n": 0}
    q = lambda p: float(np.percentile(v, p))
    return {"n": int(v.size), "valid_frac": float(v.size) / z.size,
            "p10": q(10), "p25": q(25), "p50": q(50), "p75": q(75),
            "p90": q(90), "mean": float(v.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="depth_two_cameras", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/twocam", metavar="DIR")
    ap.add_argument("--frames", type=int, default=3,
                    help="pairs per camera; the LAST is drawn, all are tabulated")
    ap.add_argument("--warmup", type=int, default=45,
                    help="frames to discard per grab so auto-exposure settles")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--domain", choices=("obs", "metric"), default="metric",
                    help="metric (default here) is a LINEAR millimetre ramp, "
                         "which is the easiest axis on which to judge 'same "
                         "distance, same colour' by eye; obs is the inverse-"
                         "depth tensor the policy actually consumes")
    ap.add_argument("--cmap", choices=sorted(CMAPS), default="turbo")
    ap.add_argument("--roi", type=float, default=0.5,
                    help="centre fraction of the frame the statistics use "
                         "(default: %(default)s). The two cameras have "
                         "different fields of view; only the middle is looking "
                         "at the same thing.")
    ap.add_argument("--c3-scale", type=float, default=1.0, metavar="K",
                    help="multiply the C3's depth by K before drawing "
                         "(default: %(default)s = untouched). A HYPOTHESIS "
                         "KNOB, not a calibration: the C3's factory "
                         "calibration is the vendor's UNDERWATER one, so in "
                         "air its depth should read long by roughly the index "
                         "of refraction. depth = fx*B/disparity, so an fx that "
                         "is wrong by a constant factor makes the depth wrong "
                         "by the SAME constant factor — which is why a scalar "
                         "is the right shape for the refraction hypothesis "
                         "specifically, and is NOT the right shape for the "
                         "device-stereo error, which grows with range "
                         "(KNOWN_ISSUES 2026-08-24). Try 1.0 and 1/1.33 = "
                         "0.752 and see which lines the two pictures up. The "
                         "value is burned into the image and written to the "
                         "JSON; a run with K != 1.0 is not a measurement of "
                         "the camera.")
    ap.add_argument("--oak-scale", type=float, default=1.0, metavar="K",
                    help="the same knob for the OAK-D-W (default: %(default)s). "
                         "Its calibration is in AIR, so in air it should need "
                         "nothing and IN WATER it is the one that reads long.")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--scale-img", type=int, default=2)
    a = ap.parse_args(argv)

    # ---- DEPTHAI FIRST, BEFORE cv2 AND TORCH -------------------------------
    # ORDER IS LOAD-BEARING, and getting it wrong looks like a hardware fault.
    # If cv2 and the FoundationStereo module (which drags in torch and a CUDA
    # context) are imported before XLink is initialised, the USB camera stops
    # appearing in `getAllAvailableDevices()` FOR THAT PROCESS ONLY — for at
    # least 30 s, while a second python started alongside sees it immediately
    # [측정 2026-09-06: 20 retries x 1.5 s inside the tool saw only the PoE C3;
    # a bare interpreter listed both throughout]. The PoE device is unaffected,
    # which is what makes it read as "the USB camera is broken".
    #
    # So: import depthai, enumerate, and OPEN BOTH DEVICES before anything
    # heavy is imported. The handles are then held for the whole run — opening
    # a throwaway probe to read the calibration and closing it empties the
    # enumeration too.
    import depthai as dai
    print("cameras")
    for k in range(20):
        seen = {d.getMxId(): d.state.name
                for d in dai.Device.getAllAvailableDevices()}
        if LAND_MXID in seen and C3_MXID in seen:
            break
        print(f"  waiting for both cameras to enumerate ({k}): {seen or 'none'}")
        time.sleep(1.5)
    d_umi = _open(LAND_MXID, a.fps)
    print(f"  OAK-D-W : open  ({LAND_MXID})")
    d_c3 = _open(C3_MXID, a.fps)
    print(f"  C3      : open  ({C3_MXID})")

    # ---- now the heavy imports, with both cameras already in hand ----------
    import cv2
    from rov_gui.perception.fstereo import FStereoSession
    from c3_camera.host_depth import StereoRig

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("rigs")
    umi_rect = load_training_rectifier(a.dataset, MONO)
    print(f"  OAK-D-W : umi_dataset.Rectifier(alpha=0.0)  fx {umi_rect.fx:.4f} px  "
          f"baseline {umi_rect.baseline_m * 1000:.3f} mm   [calibration IN AIR]")
    c3_rig = StereoRig.from_calibration_handler(d_c3.readCalibration(),
                                                mono_size=MONO, alpha=0.0)
    print(f"  C3      : StereoRig.from_calibration_handler(alpha=0.0)  "
          f"fx {c3_rig.fx_rect:.4f} px  baseline {c3_rig.baseline_mm:.3f} mm   "
          f"[calibration UNDERWATER]")
    print(f"  fx ratio C3/OAK = {c3_rig.fx_rect / umi_rect.fx:.4f}, "
          f"baseline ratio = {c3_rig.baseline_mm / (umi_rect.baseline_m * 1000):.4f}")

    fs = FStereoSession(iters=ITERS, scale=SCALE)
    t0 = time.time()
    fs.start_async(on_log=lambda lvl, m: print(f"  [{lvl}] {m}"))
    while not fs.ready and not fs.error and time.time() - t0 < 600:
        time.sleep(0.2)
    if fs.error:
        print(f"FoundationStereo failed: {fs.error}", file=sys.stderr)
        return 3
    print(f"  FoundationStereo iters {ITERS} scale {SCALE} ready in "
          f"{time.time() - t0:.1f} s  (ONE session serves both cameras)")

    def depth_of(dev, kind):
        l, r = _grab(dev, a.warmup)
        if kind == "umi":
            rl, rr = umi_rect.rectify(l, r)
            disp = fs.disparity(rl, rr)
            return umi_rect.disparity_to_depth_mm(disp) * float(a.oak_scale), rl
        rl = cv2.remap(l, *c3_rig.map_left, interpolation=cv2.INTER_LINEAR)
        rr = cv2.remap(r, *c3_rig.map_right, interpolation=cv2.INTER_LINEAR)
        disp = fs.disparity(rl, rr)
        z = np.zeros(disp.shape, np.float32)
        good = disp > 0.1
        z[good] = c3_rig.fx_rect * c3_rig.baseline_mm / disp[good]
        return z * float(a.c3_scale), rl

    print(f"\ncapturing {a.frames} pair(s) per camera "
          f"(warm-up {a.warmup} frames each)")
    rows = []
    last = {}
    try:
        for k in range(max(1, a.frames)):
            zu, mu = depth_of(d_umi, "umi")
            zc, mc = depth_of(d_c3, "c3")
            su, sc = _stats(zu / 1000.0, a.roi), _stats(zc / 1000.0, a.roi)
            # Where the near-field blob sits, the same number
            # `depth_compare stats` reports for the training obs (0.895 there).
            # A camera mounted upside down relative to the demonstrations puts
            # it at the wrong end, and that is invisible in a depth histogram.
            zz_u = np.where(zu > 0, zu / 1000.0, np.nan)
            zz_c = np.where(zc > 0, zc / 1000.0, np.nan)
            su["near_row"], sc["near_row"] = near_row(zz_u), near_row(zz_c)
            rows.append((su, sc))
            last = {"zu": zu, "zc": zc, "mu": mu, "mc": mc}
            if su.get("n") and sc.get("n"):
                print(f"  [{k}] centre {100 * a.roi:.0f}%  "
                      f"OAK p50 {su['p50']:.3f} m   C3 p50 {sc['p50']:.3f} m   "
                      f"ratio C3/OAK {sc['p50'] / su['p50']:.3f}   "
                      f"near-row OAK {su['near_row']:.3f} / C3 {sc['near_row']:.3f} "
                      f"(land training = 0.895)")
            else:
                print(f"  [{k}] no valid pixels in the centre box")
    finally:
        for d in (d_umi, d_c3):
            try:
                d.close()
            except Exception:                                    # noqa: BLE001
                pass

    good = [(u, c) for u, c in rows if u.get("n") and c.get("n")]
    ratio = float(np.median([c["p50"] / u["p50"] for u, c in good])) if good else float("nan")

    # ---- the picture: ONE renderer, ONE rule, both cameras -------------------
    def panel(z_mm, title, sub):
        z = z_mm.astype(np.float32) / 1000.0
        valid = z_mm > 0
        t = palette_pos(np.where(valid, z, Z_FAR_M), a.domain, Z_NEAR_M, Z_FAR_M)
        return render(t, valid, title=title, domain=a.domain, cmap=a.cmap,
                      z_near=Z_NEAR_M, z_far=Z_FAR_M, subtitle=sub,
                      scale=a.scale_img)

    def scale_note(k, medium):
        return (f"calib {medium}" if abs(k - 1.0) < 1e-9
                else f"calib {medium} | DEPTH x{k:g} APPLIED")
    pu = panel(last["zu"], "OAK-D-W  (land training camera)",
               scale_note(a.oak_scale, "IN AIR"))
    pc = panel(last["zc"], "C3  (deployment camera)",
               scale_note(a.c3_scale, "UNDERWATER"))
    sheet = side_by_side(pu, pc)
    cv2.imwrite(str(out / "two_cameras_depth.png"), sheet)

    mono = np.hstack([cv2.cvtColor(last["mu"], cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(last["mc"], cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(str(out / "two_cameras_mono.png"), mono)

    meta = {
        "colour_rule": colour_rule_json(a.domain, a.cmap, Z_NEAR_M, Z_FAR_M),
        "note": "ONE FoundationStereo session, ONE renderer, ONE colour rule. "
                "Any colour difference in the picture is a difference in the "
                "DEPTH, not in the drawing.",
        "fstereo": fs.describe(),
        "roi_centre_fraction": a.roi,
        "cameras": {
            "oak_d_w": {"mxid": LAND_MXID, "calibration_medium": "air",
                        "rectifier": "umi_dataset.Rectifier(alpha=0.0)",
                        "fx_rect_px": float(umi_rect.fx),
                        "baseline_mm": float(umi_rect.baseline_m * 1000)},
            "c3": {"mxid": C3_MXID, "calibration_medium": "water",
                   "rectifier": "c3_camera.host_depth.StereoRig(alpha=0.0)",
                   "fx_rect_px": float(c3_rig.fx_rect),
                   "baseline_mm": float(c3_rig.baseline_mm)},
        },
        "frames": [{"oak": u, "c3": c} for u, c in rows],
        "median_ratio_c3_over_oak": ratio,
        "ratio_is_a_calibration_measurement": bool(np.isfinite(ratio)
                                                   and 0.6 < ratio < 1.7),
        "near_row_land_training_reference": 0.895,
        "applied_scales": {"c3": float(a.c3_scale), "oak_d_w": float(a.oak_scale),
                           "note": "multipliers applied to the DEPTH before "
                                   "drawing and before these statistics. 1.0 "
                                   "means untouched. Anything else makes this "
                                   "run a hypothesis test, not a measurement "
                                   "of the cameras."},
        "expectation": "in AIR the C3 should read LONG by roughly the index of "
                       "refraction (~1.33x) because its factory calibration is "
                       "the vendor's underwater one; the OAK-D-W should read "
                       "true. Underwater the roles swap. Nothing here corrects "
                       "anything.",
    }
    (out / "two_cameras.json").write_text(json.dumps(meta, indent=2, default=str))

    if abs(a.c3_scale - 1.0) > 1e-9 or abs(a.oak_scale - 1.0) > 1e-9:
        print(f"\nSCALES APPLIED  C3 x{a.c3_scale:g}  OAK x{a.oak_scale:g}  "
              f"— this run is a hypothesis test, not a measurement")
    print(f"\nmedian ratio  C3 / OAK-D-W = {ratio:.3f}"
          f"   (in air, ~1.33 would be the refractive-index story)")
    if np.isfinite(ratio) and not (0.6 < ratio < 1.7):
        print("\n  *** THIS RATIO IS NOT A CALIBRATION MEASUREMENT ***\n"
              "  A refractive-index story lives between about 0.75 and 1.33. A\n"
              "  ratio this far outside it means the two cameras are not looking\n"
              "  at the SAME SURFACE — different fields of view, different\n"
              "  mounting, or one of them staring at something near while the\n"
              "  other sees down the room. Aim both at one flat surface that\n"
              "  FILLS both frames, at a taped distance, and re-run. Until then\n"
              "  this number measures the framing, not the cameras.")
    if good:
        nu = float(np.median([u.get("near_row", np.nan) for u in
                              [x for x, _ in rows] if np.isfinite(x.get("near_row", np.nan))] or [np.nan]))
        nc = float(np.median([c.get("near_row", np.nan) for _, c in rows
                              if np.isfinite(c.get("near_row", np.nan))] or [np.nan]))
        print(f"\nnear-field row (0 = top, 1 = bottom; land training = 0.895)")
        print(f"  OAK-D-W {nu:.3f}   C3 {nc:.3f}")
        for tag, v in (("OAK-D-W", nu), ("C3", nc)):
            if np.isfinite(v):
                ok = "matches the training layout" if v > 0.6 else \
                     "UPSIDE DOWN relative to the demonstrations"
                print(f"    {tag:8s} {ok}")
    if good and np.isfinite(ratio) and ratio > 0:
        need = float(a.c3_scale) / ratio
        print(f"to line the two up as drawn, pass  --c3-scale {need:.3f}"
              f"   (1/1.33 = 0.752 is the refraction prediction)")
    print(f"wrote {out / 'two_cameras_depth.png'}   (same colour rule both sides)")
    print(f"wrote {out / 'two_cameras_mono.png'}    (what each camera saw)")
    print(f"wrote {out / 'two_cameras.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
