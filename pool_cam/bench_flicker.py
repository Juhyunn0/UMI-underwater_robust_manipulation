#!/usr/bin/env python3
"""Rolling horizontal bands (light flicker) per anti-flicker setting, measured.

    python pool_cam/bench_flicker.py --device cam0=/dev/video0 --device cam1=/dev/video2
    python pool_cam/bench_flicker.py --out pool_cam/bench_out/flicker_<date>.txt

For each camera and each power-line setting (50 Hz, 60 Hz, off): let the auto
exposure settle 1.5 s, then take 45 frames and report

  rolling-band   std of each row's brightness from frame to frame, after the
                 frame's overall brightness is taken out -- what a mismatched
                 anti-flicker setting produces (bands that crawl up or down);
                 the scene must hold still, so point the cameras at something
                 that does
  row-stripe     std of one frame's row profile around its 101-row trend --
                 fixed stripes, from the sensor or the scene

The camera is left at --leave (default 60) afterwards. Needs the cameras free
(close ./c3 cctv / the station first).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from pool_cam.pool_cam import (POOL_MODELS, is_pool, parse_device, set_power_line,  # noqa: E402
                               usb_cameras)


def measure(dev, settle=45, take=45):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        return None
    for _ in range(30):
        cap.read()
    out = {}
    for hz in ("50", "60", "off"):
        err = set_power_line(dev, hz)
        if err:
            out[hz] = err
            continue
        for _ in range(settle):
            cap.read()
        rows = []
        for _ in range(take):
            ok, f = cap.read()
            if ok:
                rows.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32).mean(axis=1))
        R = np.array(rows)
        resid = R - R.mean(axis=0)
        resid -= resid.mean(axis=1, keepdims=True)
        k = 101
        prof = R[-1]
        trend = np.convolve(np.pad(prof, k // 2, mode="edge"), np.ones(k) / k, "valid")
        out[hz] = (float(resid.std()), float((prof - trend).std()), float(R.mean()), len(rows))
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", action="append", default=[], metavar="NAME=DEV")
    ap.add_argument("--leave", default="60", choices=("50", "60", "off"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.device:
        specs = [parse_device(d, i) for i, d in enumerate(a.device)]
    else:
        specs = [(f"cam{i}", dev) for i, (dev, model, _) in enumerate(
            (c for c in usb_cameras() if is_pool(c[1], POOL_MODELS)))]
    lines = [f"# pool_cam/bench_flicker.py  {time.strftime('%Y-%m-%d %H:%M:%S')}",
             "# rolling-band = frame-to-frame row-brightness std (crawling bands); "
             "row-stripe = fixed row pattern; 1080p30 MJPEG, 45 frames after 1.5 s settle"]
    models = {dev: (model, usb) for dev, model, usb in usb_cameras()}
    for name, dev in specs:
        res = measure(dev)
        model, usb = models.get(dev, ("?", "?"))
        lines.append(f"{name}  {dev}  {usb}  {model.strip()}")
        if res is None:
            lines.append("   could not open (busy?)")
            continue
        for hz, r in res.items():
            if isinstance(r, str):
                lines.append(f"   {hz:>3s}  not settable: {r}")
            else:
                lines.append(f"   {hz:>3s}  rolling-band {r[0]:5.2f}   row-stripe {r[1]:5.2f}   "
                             f"level {r[2]:5.1f}   frames {r[3]}")
        set_power_line(dev, a.leave)
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(report)


if __name__ == "__main__":
    main()
