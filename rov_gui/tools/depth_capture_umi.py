#!/usr/bin/env python3
"""
depth_capture_umi.py — capture live depth through the LAND TRAINING pipeline, verbatim.

Why this exists
---------------
On 2026-09-06 the OAK-D-W that collected the land demonstrations
(``mxid 14442C10716DBCD600``) was mounted on the ROV alongside the C3. That makes a
land-vs-water depth comparison far cleaner than it could ever be through the C3,
because the *instrument* stops being a variable:

    same sensors, same EEPROM calibration, same rectification (alpha=0.0, fx
    249.5656 px @ 640x400), same FoundationStereo checkpoint, same iters, same
    scale, same warp, same crop, same resize, same normalisation.

What is left differing is the scene, the medium, and where the rig is bolted. That
is the comparison Max asked for on 2026-09-04, and until this camera moved onto the
vehicle it was not obtainable: the C3 path differs in sensor, calibration medium,
optics and FoundationStereo settings all at once.

So this tool does NOT invent a pipeline. Every stage is the object the training
data went through:

    dai MonoCamera CAM_B/CAM_C 640x400   the same sockets record.py opens
    umi_dataset.Rectifier(alpha=0.0)     THE training rectifier, from the dataset's
                                         own checkout and its own calibration.json
    FoundationStereo                     rov_gui.perception.fstereo, at the TRAINING
                                         settings (iters 16, scale 1.0) by default
    rect.disparity_to_depth_mm           the training disparity->mm, its own fx
    umi_handheld.warp.WarpStage          the training warp onto C3 CAM_B optics
    centre crop -> 224 nearest -> normalise_depth(0.20, 3.00)
                                         build_dp_depth_zarr.py's per-frame loop

and writes the ``policy_obs/`` layout ``depth_compare`` already reads, so::

    python -m rov_gui.tools.depth_capture_umi --out /tmp/rovcap --frames 40
    python -m rov_gui.tools.depth_compare pair --run /tmp/rovcap --episode 12 \
           --out /tmp/pair

MEDIUM. This camera's calibration is IN AIR. The C3's factory calibration is
underwater (vendor-confirmed; see the project memory). So underwater this rig's
depth reads long by roughly the index of refraction unless a re-derived
calibration is supplied, the same ~1.33x the C3 shows in air but in the opposite
direction. ``medium`` is recorded in the meta and printed at start; it is a LABEL,
not a correction, and nothing here rescales anything.

This tool never commands the vehicle. It opens a camera and writes files.
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

#: Where the land demonstrations, their calibration and the training checkout live.
DEFAULT_DATASET = Path("/home/bdml/Desktop/data collection")

#: The unit that collected the land data. Pinned by default, because both this
#: camera and the C3 answer DepthAI discovery and ``dai.Device()`` with no
#: argument takes whichever it enumerates first — a mistake that has already
#: cost one benchmark run (umi_handheld/record.py:find_device).
LAND_MXID = "14442C10716DBCD600"

#: The obs recipe, frozen with the training store.
Z_NEAR_M, Z_FAR_M, OUT_RES = 0.20, 3.00, 224
WARP_SIZE = (640, 400)

#: The training depth settings [측정: ~/Desktop/data collection/dataset_depth.zarr.zip
#: .zattrs['depth_source'] -> iters 16, scale 1.0]. Deliberately NOT the station's
#: --policy defaults (iters 8, scale 0.75), which trade instrument parity for rate.
TRAIN_ITERS, TRAIN_SCALE = 16, 1.0


def build_pipeline(fps: float):
    """CAM_B / CAM_C mono at 640x400, raw. The same sockets and resolution the
    demonstrations were recorded at; no on-device StereoDepth is created, because
    the disparity comes from FoundationStereo on the HOST."""
    import depthai as dai
    p = dai.Pipeline()
    outs = {}
    for name, sock in (("left", dai.CameraBoardSocket.CAM_B),
                       ("right", dai.CameraBoardSocket.CAM_C)):
        m = p.create(dai.node.MonoCamera)
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setBoardSocket(sock)
        m.setFps(float(fps))
        x = p.create(dai.node.XLinkOut)
        x.setStreamName(name)
        m.out.link(x.input)
        outs[name] = x
    return p


def open_device(pipeline, mxid: str | None):
    import depthai as dai
    if not mxid:
        return dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
    found = {d.getMxId(): d for d in dai.Device.getAllAvailableDevices()}
    if mxid not in found:
        raise RuntimeError(
            f"device {mxid} is not reachable. Visible: "
            + (", ".join(f"{k} ({v.protocol.name})" for k, v in found.items())
               or "none")
            + ". Pass --any-device to accept whatever is attached, knowing the "
              "rectifier's calibration then may not describe it.")
    return dai.Device(pipeline, found[mxid], maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)


def load_training_rectifier(dataset: Path, size=(640, 400)):
    """THE training rectifier, from the dataset's own checkout and calibration.

    Imported rather than reimplemented so the live pair cannot drift from the
    frames the network was trained on: this is the same class, the same
    ``alpha=0.0``, and the same ``calibration.json`` that
    ``replay_foundation_stereo.py`` used.
    """
    import types
    sys.path.insert(0, str(dataset / "UMI_Underwater"))
    # `umi_dataset` imports PyAV at module scope for its video reader, and the
    # station env has no `av`. `Rectifier` uses only cv2 and numpy, so the name
    # is OWNED FOR THE DURATION OF THE IMPORT and then handed back — the same
    # move, for the same reason, as fstereo._import_upstream. Importing the
    # real class matters more than the missing decoder: a reimplementation here
    # could drift from the rectification the training depth actually lives on.
    injected = "av" not in sys.modules
    if injected:
        sys.modules["av"] = types.ModuleType("av")
    try:
        from umi_dataset import Rectifier                      # noqa: E402
    finally:
        if injected:
            sys.modules.pop("av", None)
    calib = json.loads((dataset / "calibration.json").read_text())
    return Rectifier(calib, size, alpha=0.0)


def compare_live_calibration(rect, device, size=(640, 400)) -> dict:
    """Live EEPROM vs the stored calibration the rectifier was built from.

    A silent mismatch here would mean the live pixels are rectified by another
    unit's numbers, which is exactly the failure the mxid pin exists to prevent
    and which no downstream stage could detect.
    """
    import cv2
    try:
        cal = device.readCalibration()
        w, h = size
        K = np.asarray(cal.getCameraIntrinsics(
            __import__("depthai").CameraBoardSocket.CAM_B, w, h), np.float64)
    except Exception as e:                                     # noqa: BLE001
        return {"checked": False, "why": f"{type(e).__name__}: {e}"}
    d = float(np.max(np.abs(K - rect.K_left)))
    return {"checked": True, "max_abs_dK_px": d,
            "ok": bool(d < 1e-2),
            "K_live": np.round(K, 4).tolist(),
            "K_stored": np.round(rect.K_left, 4).tolist()}


def source_model_from_rectifier(rect, dataset: Path, size) -> "object":
    """The pinhole the FoundationStereo depth lives on, as a CameraModel.

    Verbatim the dict ``build_dp_depth_zarr.rectified_left_model`` builds, from
    the same ``Rectifier`` instance, so the live warp reads the same pixels the
    training warp did.
    """
    from umi_handheld.camera_model import CameraModel
    P1 = np.asarray(rect.P1, dtype=np.float64)
    w, h = (int(v) for v in size)
    cfg = {
        "schema": "umi_camera_model/1",
        "name": "oakd_oak_d_w_rectified_left_alpha0",
        "role": "source", "medium": "air", "verified": True, "rectified": True,
        "image_size": [w, h], "model": "opencv_rational",
        "intrinsics": {"fx": float(P1[0, 0]), "fy": float(P1[1, 1]),
                       "cx": float(P1[0, 2]), "cy": float(P1[1, 2])},
        "distortion": [0.0] * 8,
        "stereo": {"baseline_m": float(rect.baseline_m)},
        "provenance": {
            "derived_from": "umi_dataset.Rectifier(calibration.json, size, alpha=0.0)",
            "same_object_as": "replay_foundation_stereo.py's rectification",
            "device_mxid": LAND_MXID,
        },
    }
    return CameraModel(cfg, dataset / "calibration.json")


def frozen_source_model(dataset: Path) -> dict | None:
    """``camera_model_source`` out of the training store's ``.zattrs``.

    Read with plain ``zipfile`` so this works in the station env, which has no
    zarr: a zarr v2 store is JSON plus chunk files, and the attributes are one
    of the JSON members.
    """
    import zipfile
    for name in ("dataset_depth.zarr.zip", "dataset_depth.zarr"):
        p = dataset / name
        try:
            if p.is_file():
                with zipfile.ZipFile(p) as z:
                    return json.loads(z.read(".zattrs"))["camera_model_source"]
            if p.is_dir():
                return json.loads((p / ".zattrs").read_text())["camera_model_source"]
        except Exception:                                      # noqa: BLE001
            continue
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="depth_capture_umi", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="capture folder; <DIR>/policy_obs/ is written, which is "
                         "exactly what `depth_compare water|pair --run <DIR>` reads")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="the land dataset, for calibration.json and the "
                         "UMI_Underwater checkout (default: %(default)s)")
    ap.add_argument("--mxid", default=LAND_MXID,
                    help="device to pin (default: the unit that collected the "
                         "land demonstrations)")
    ap.add_argument("--any-device", action="store_true")
    ap.add_argument("--medium", choices=("air", "water"), default="air",
                    help="a LABEL for the record, never a correction. This "
                         "camera's calibration is in-air, so underwater its "
                         "depth reads long by roughly the refractive index "
                         "unless a re-derived calibration is supplied.")
    ap.add_argument("--iters", type=int, default=TRAIN_ITERS)
    ap.add_argument("--scale", type=float, default=TRAIN_SCALE)
    ap.add_argument("--target-model", default="configs/target_camera_underwater.yaml")
    ap.add_argument("--no-warp", action="store_true",
                    help="skip the warp onto C3 optics. Only for inspecting the "
                         "raw rectified-left depth; the obs would then NOT be "
                         "the tensor the policy was trained on.")
    ap.add_argument("--note", default="", help="free text into meta.json")
    a = ap.parse_args(argv)

    import cv2
    import depthai as dai                                      # noqa: F401
    from umi_handheld.build_zarr import normalise_depth
    from umi_handheld.camera_model import CameraModel
    from umi_handheld.warp import WarpStage
    from rov_gui.perception.fstereo import FStereoSession

    out = Path(a.out)
    obs_dir, depth_dir = out / "policy_obs" / "obs", out / "policy_obs" / "depth"
    obs_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset   : {a.dataset}")
    rect = load_training_rectifier(a.dataset)
    print(f"rectifier : alpha=0.0 @ 640x400  fx {rect.fx:.4f} px  "
          f"baseline {rect.baseline_m * 1000:.3f} mm   (THE training rectifier)")

    # The source optical model, built from the SAME Rectifier object above and
    # then CHECKED AGAINST THE TRAINING STORE'S OWN FROZEN RECORD. Importing
    # build_dp_depth_zarr.rectified_left_model would be the obvious way to
    # avoid drift, but that module imports zarr/numcodecs at module scope and
    # the station env has neither. Asserting against
    # `dataset_depth.zarr.zip .zattrs['camera_model_source']` is the stronger
    # check anyway: it compares against the numbers the shipped dataset was
    # actually built with, not against code that could have changed since.
    src_cam = source_model_from_rectifier(rect, a.dataset, (640, 400))
    frozen = frozen_source_model(a.dataset)
    if frozen:
        for k in ("fx", "fy", "cx", "cy"):
            got, want = float(getattr(src_cam, k)), float(frozen[k])
            if abs(got - want) > 1e-6:
                print(f"REFUSING: live rectifier {k}={got!r} but the training "
                      f"store was built with {k}={want!r}. The warp would read "
                      f"different pixels than the policy was trained on.",
                      file=sys.stderr)
                return 3
        print(f"parity    : source optics match the training store's "
              f"camera_model_source exactly (fx/fy/cx/cy)")
    else:
        print("parity    : could not read the training store's "
              "camera_model_source; source optics UNVERIFIED", file=sys.stderr)
    tgt_path = a.target_model
    if not Path(tgt_path).is_absolute() and not Path(tgt_path).exists():
        tgt_path = str(REPO / tgt_path)
    tgt_cam = CameraModel.load(tgt_path)
    print(f"source    : {src_cam.name} fx {src_cam.fx:.4f} @ "
          f"{src_cam.width}x{src_cam.height} ({src_cam.medium})")
    print(f"target    : {tgt_cam.name} fx {tgt_cam.fx:.2f} @ "
          f"{tgt_cam.width}x{tgt_cam.height} ({tgt_cam.medium})")
    if abs(src_cam.fx - rect.fx) > 1e-3:
        print(f"REFUSING: the source model's fx {src_cam.fx} disagrees with the "
              f"live rectifier's {rect.fx}; the warp would read the wrong "
              f"pixels.", file=sys.stderr)
        return 3

    stage = None
    if not a.no_warp:
        stage = WarpStage(src_cam, tgt_cam, WARP_SIZE,
                          {"depth_interpolation": "nearest",
                           "mono_interpolation": "linear"},
                          source_size=(640, 400))
        print(f"warp      : 640x400 -> {WARP_SIZE}  coverage "
              f"{100 * stage.coverage:.2f}%  behind-camera rays {stage.behind}")

    print(f"fstereo   : iters {a.iters}, scale {a.scale}  "
          f"({'TRAINING settings' if (a.iters, a.scale) == (TRAIN_ITERS, TRAIN_SCALE) else 'NOT the training settings'})")
    fs = FStereoSession(iters=int(a.iters), scale=float(a.scale))
    t0 = time.time()
    fs.start_async(on_log=lambda lvl, m: print(f"            [{lvl}] {m}"))
    while not fs.ready and not fs.error and time.time() - t0 < 600:
        time.sleep(0.2)
    if fs.error:
        print(f"FoundationStereo failed to load: {fs.error}", file=sys.stderr)
        return 3
    print(f"            loaded in {time.time() - t0:.1f} s")

    pipeline = build_pipeline(a.fps)
    print(f"device    : opening {'any' if a.any_device else a.mxid}")
    rows, cal_check = [], {}
    t_start = time.time()
    with open_device(pipeline, None if a.any_device else a.mxid) as dev:
        cal_check = compare_live_calibration(rect, dev)
        if cal_check.get("checked"):
            verdict = "OK" if cal_check["ok"] else "MISMATCH"
            print(f"calib     : live EEPROM vs stored calibration.json "
                  f"max|dK| {cal_check['max_abs_dK_px']:.4f} px  {verdict}")
            if not cal_check["ok"]:
                print("            the live unit is NOT the one calibration.json "
                      "describes; frames would be rectified by another camera's "
                      "numbers.", file=sys.stderr)
                return 3
        else:
            print(f"calib     : live EEPROM unreadable ({cal_check.get('why')}) — "
                  f"proceeding on the stored calibration")

        ql = dev.getOutputQueue("left", 4, False)
        qr = dev.getOutputQueue("right", 4, False)
        seq = 0
        print(f"capturing {a.frames} frames...")
        while seq < a.frames:
            fl, fr = ql.get(), qr.get()
            left, right = fl.getCvFrame(), fr.getCvFrame()
            # THE training chain, in order.
            rl, rr = rect.rectify(left, right)
            disp = fs.disparity(rl, rr)
            depth_mm = rect.disparity_to_depth_mm(disp)        # float32 mm, 0 = invalid
            w = stage.apply_depth(depth_mm) if stage is not None else depth_mm
            s = min(w.shape[:2])
            y0, x0 = (w.shape[0] - s) // 2, (w.shape[1] - s) // 2
            w = w[y0:y0 + s, x0:x0 + s]
            w = cv2.resize(w, (OUT_RES, OUT_RES), interpolation=cv2.INTER_NEAREST)
            v = normalise_depth(w, Z_NEAR_M, Z_FAR_M)
            valid = (w > 0).astype(np.uint8) * 255
            u8 = np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)
            obs = np.stack([u8, valid, u8], axis=-1)

            png = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            on, dn = f"obs/{seq:06d}.png", f"depth/{seq:06d}.png"
            if not cv2.imwrite(str(out / "policy_obs" / on), obs, png):
                print(f"write failed: {on}", file=sys.stderr)
                break
            cv2.imwrite(str(out / "policy_obs" / dn),
                        np.clip(np.rint(w), 0, 65535).astype(np.uint16), png)
            rows.append((seq, time.time(), on, dn))
            fin = float(np.count_nonzero(valid)) / valid.size
            if seq % 5 == 0:
                z = w[w > 0] / 1000.0
                print(f"  {seq:3d}  valid {100 * fin:5.1f}%  "
                      f"median {np.median(z) if z.size else float('nan'):.3f} m")
            seq += 1

    idx = out / "policy_obs" / "index.csv"
    idx.write_text("seq,t_capture,obs_png,depth_png\n"
                   + "".join(f"{s},{t:.6f},{o},{d}\n" for s, t, o, d in rows))
    meta = {
        "schema": "rov_gui/policy_obs_recording/1",
        "written_by": "rov_gui/tools/depth_capture_umi.py",
        "t_start_wall": t_start,
        "duration_s": time.time() - t_start,
        "counters": {"offered": len(rows), "written": len(rows), "dropped_full": 0,
                     "dropped_budget": 0, "dropped_disabled": 0,
                     "dropped_no_writer": 0, "write_errors": 0,
                     "write_errors_writer": 0},
        "counts_complete": True,
        "max_frames": int(a.frames),
        "obs_png": "224x224x3 uint8, channels (inverse_depth, validity, inverse_depth)",
        "depth_png": "224x224 uint16 millimetres AFTER warp+crop+resize, 0 = no measurement",
        "read_with": "python -m rov_gui.tools.depth_compare water --run <run>",
        "obs": {"z_near_m": Z_NEAR_M, "z_far_m": Z_FAR_M, "out_res": OUT_RES,
                "crop": {"y0": 0, "x0": 120, "size": 400},
                "grid_kind": "umi_rect_left_alpha0", "coverage":
                    (float(stage.coverage) if stage is not None else None)},
        "builder": {
            "z_near_m": Z_NEAR_M, "z_far_m": Z_FAR_M, "out_res": OUT_RES,
            "grid_kind": "umi_rect_left_alpha0",
            "pipeline": "THE land training chain: umi_dataset.Rectifier(alpha=0.0) "
                        "-> FoundationStereo -> rect.disparity_to_depth_mm -> "
                        "WarpStage(nearest) -> centre crop -> 224 nearest -> "
                        "normalise_depth",
            "source_model": src_cam.provenance(),
            "target_model": tgt_cam.provenance(),
            "warp_applied": stage is not None,
            "rectifier": {"alpha": 0.0, "size": [640, 400], "fx_px": float(rect.fx),
                          "baseline_m": float(rect.baseline_m)},
            "fstereo": fs.describe(),
            "live_calibration_check": cal_check,
        },
        "extra": {
            "device_mxid": (None if a.any_device else a.mxid),
            "medium": a.medium,
            "medium_note": ("this camera's calibration is IN AIR; underwater its "
                            "depth reads long by roughly the refractive index "
                            "unless a re-derived calibration is supplied. Nothing "
                            "here rescales anything — the label is a label."),
            "note": a.note,
            "training_settings": {"iters": TRAIN_ITERS, "scale": TRAIN_SCALE},
            "used_settings": {"iters": int(a.iters), "scale": float(a.scale)},
        },
    }
    (out / "policy_obs" / "meta.json").write_text(json.dumps(meta, indent=2,
                                                            default=str))
    print(f"\nwrote {len(rows)} frames to {out / 'policy_obs'}")
    print(f"compare with the land training obs, one colour rule:\n"
          f"  python -m rov_gui.tools.depth_compare pair --run {out} "
          f"--episode 12 --out /tmp/pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
