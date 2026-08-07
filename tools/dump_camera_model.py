#!/usr/bin/env python3
"""dump_camera_model.py — write a measured camera model YAML from a connected OAK.

This is the "one file changes on the day the camera arrives" step. It replaces a
synthesised model (verified: false) with the device's own factory calibration
(verified: true); no pipeline code refers to any of the numbers it writes.

Which model does it write?
--------------------------
The one that actually describes the images record.py saves — and that is NOT the raw
CAM_B calibration. record.py stores `stereo.rectifiedLeft`, which StereoDepth has already
undistorted and row-aligned. Its correct model is the rectified projection P1 from
cv2.stereoRectify, with zero distortion.

Writing raw CAM_B here would silently apply a barrel distortion that the pixels no longer
have, corrupting both the solvePnP in extract_gripper_width.py and the remap in warp.py.
The raw intrinsics are kept in the YAML under `raw_left` for reference only.

alpha
-----
cv2.stereoRectify's alpha trades field of view against black border, and it moves the
rectified focal length by a lot (measured ~16% across the range on the C3's calibration in
c3_camera/host_depth.py:446-455). -1.0 is the default here because prior analysis of the
same lens family found the device's own rectification reproduces it. The chosen value and
the resulting P1 are both recorded in the YAML.

Usage
-----
    DEPTHAI_PROTOCOL= python tools/dump_camera_model.py --out configs/source_camera_air.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "configs/source_camera_air.yaml")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--alpha", type=float, default=-1.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--medium", default="air", choices=["air", "water"])
    ap.add_argument("--role", default="source", choices=["source", "target"])
    a = ap.parse_args()

    import depthai as dai

    with dai.Device() as dev:
        mxid = dev.getMxId()
        product = dev.getDeviceName()
        calib = dev.readCalibration()
        W, H = a.width, a.height
        L, R = dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
        K_l = np.array(calib.getCameraIntrinsics(L, W, H), np.float64)
        K_r = np.array(calib.getCameraIntrinsics(R, W, H), np.float64)
        d_l = np.array(calib.getDistortionCoefficients(L), np.float64).reshape(1, -1)
        d_r = np.array(calib.getDistortionCoefficients(R), np.float64).reshape(1, -1)
        ext = np.array(calib.getCameraExtrinsics(L, R), np.float64)   # 4x4, t in cm

    Rot = ext[:3, :3]
    T = ext[:3, 3] / 100.0                                            # cm -> m
    baseline = float(np.linalg.norm(T))

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, d_l, K_r, d_r, (W, H), Rot, T, alpha=a.alpha)

    fx, fy = float(P1[0, 0]), float(P1[1, 1])
    cx, cy = float(P1[0, 2]), float(P1[1, 2])

    sys.path.insert(0, str(REPO / "calib"))
    from fov_audit import measure_fov                                 # noqa: E402
    Krect = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    fov_rect = measure_fov(Krect, np.zeros((1, 8)), W, H)
    fov_raw = measure_fov(K_l, d_l, W, H)

    name = a.name or f"oakd_{product.lower().replace('-', '_')}_rectified_left"
    lines = [
        "# Recording camera — MEASURED from the connected device, not synthesised.",
        "#",
        f"# Written by tools/dump_camera_model.py from {product} mxid {mxid}.",
        "# Every number below came off the device's factory EEPROM; none is typed by hand.",
        "#",
        "# This describes the RECTIFIED-LEFT stream, which is what record.py saves.",
        "# StereoDepth has already removed the lens distortion, so `distortion` is zero",
        "# by construction and the intrinsics are P1 from cv2.stereoRectify. The raw",
        "# CAM_B calibration is under `raw_left` for reference and is NOT what the",
        "# saved pixels obey.",
        "",
        "schema: umi_camera_model/1",
        f"name: {name}",
        f"role: {a.role}",
        f"medium: {a.medium}",
        "verified: true",
        "",
        "provenance:",
        f"  device_product: {product}",
        f"  device_mxid: {mxid}",
        "  source: on-device factory calibration (EEPROM), read via depthai readCalibration()",
        "  tool: tools/dump_camera_model.py",
        "  describes: stereo.rectifiedLeft (what record.py writes to left/)",
        "  rectification:",
        "    method: cv2.stereoRectify",
        f"    alpha: {a.alpha}",
        f"    baseline_m_from_extrinsics: {baseline:.9f}",
        f"    R1: {np.round(R1, 12).tolist()}",
        f"    P1: {np.round(P1, 9).tolist()}",
        "  measured_fov_deg:",
        f"    rectified_hfov: {fov_rect['hfov']:.4f}",
        f"    rectified_vfov: {fov_rect['vfov']:.4f}",
        f"    raw_left_hfov: {fov_raw['hfov']:.4f}",
        f"    raw_left_vfov: {fov_raw['vfov']:.4f}",
        "",
        f"image_size: [{W}, {H}]",
        "",
        "model: opencv_rational",
        "intrinsics:",
        f"  fx: {fx:.9f}",
        f"  fy: {fy:.9f}",
        f"  cx: {cx:.9f}",
        f"  cy: {cy:.9f}",
        "# zero by construction: the rectified stream is already undistorted",
        "distortion: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
        "",
        "fov_deg:",
        f"  hfov: {fov_rect['hfov']:.4f}",
        f"  vfov: {fov_rect['vfov']:.4f}",
        f"  valid_upto_half_angle: {fov_rect['valid_upto_deg']:.2f}",
        "",
        "stereo:",
        f"  baseline_m: {baseline:.9f}",
        "  left_socket: CAM_B",
        "  right_socket: CAM_C",
        "  right_shares_model: false",
        "",
        "rectified: true",
        "",
        "# The raw sensor calibration. Kept so the unrectified grid can be reconstructed;",
        "# nothing in the pipeline reads it.",
        "raw_left:",
        f"  fx: {K_l[0,0]:.9f}",
        f"  fy: {K_l[1,1]:.9f}",
        f"  cx: {K_l[0,2]:.9f}",
        f"  cy: {K_l[1,2]:.9f}",
        f"  distortion: {np.round(d_l.ravel(), 10).tolist()}",
        "raw_right:",
        f"  fx: {K_r[0,0]:.9f}",
        f"  fy: {K_r[1,1]:.9f}",
        f"  cx: {K_r[0,2]:.9f}",
        f"  cy: {K_r[1,2]:.9f}",
        f"  distortion: {np.round(d_r.ravel(), 10).tolist()}",
        "",
    ]
    a.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"device        : {product}  mxid {mxid}")
    print(f"baseline      : {baseline*1000:.2f} mm (from extrinsics)")
    print(f"raw CAM_B     : fx {K_l[0,0]:.2f}  HFOV {fov_raw['hfov']:.2f}d  "
          f"dist n={d_l.size}")
    print(f"rectified P1  : fx {fx:.2f}  cx {cx:.2f} cy {cy:.2f}  "
          f"HFOV {fov_rect['hfov']:.2f}d  (alpha={a.alpha})")
    print(f"wrote         : {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
