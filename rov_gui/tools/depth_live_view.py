#!/usr/bin/env python3
"""
depth_live_view.py — both cameras, live, one FoundationStereo, one colour rule.

    ./c3 depth-live

Two panes side by side, updating continuously: the OAK-D-W that recorded the
land demonstrations on the left, the C3 that flies the vehicle on the right.
ONE FoundationStereo session runs both, at the training settings, and ONE
renderer draws both on the same fixed 0.20-3.00 m scale with no auto-range and
no histogram equalisation. So a colour difference on screen is a difference in
the DEPTH, not in the drawing.

The point is the scale knob
---------------------------
The C3's factory calibration is the vendor's UNDERWATER one, so in air its
depth should read long by roughly the index of refraction; the OAK-D-W's is in
air and should read true there. Underwater the roles swap. ``[`` and ``]`` walk
the C3's depth multiplier live, so you can watch the two pictures slide into
agreement and read the factor off the screen instead of arguing about it.

``depth = fx * baseline / disparity``, so a focal length wrong by a constant
factor makes the depth wrong by the SAME constant factor — which is why a
scalar is the right shape for the refraction hypothesis. It is NOT the right
shape for the C3's on-device stereo error, which grows with range
(KNOWN_ISSUES 2026-08-24). This viewer only ever shows the FoundationStereo
path, so the scalar is the honest knob here.

Aim both cameras at ONE flat surface that FILLS both frames before believing
the ratio. Different framing produces a ratio that measures the framing.

Keys
----
    [ ]   C3 depth scale  -/+ 1%          { }   -/+ 10%
    , .   OAK depth scale -/+ 1%
    1     C3 scale to 1/1.33 = 0.752 (the refraction prediction)
    0 r   reset both scales to 1.0
    m     metric <-> obs domain      c     next colormap
    f     freeze / unfreeze          s     save a PNG next to --out
    q Esc quit

Why Qt and not cv2.imshow
-------------------------
``cv2.imshow`` aborts in this environment: cv2 ships its own Qt platform
plugins and they lose to PyQt5's, which is the documented xcb crash in
``rov_gui/qt.py``'s module docstring. The station already runs on PyQt5, so
that is the path that works here.

This tool opens two cameras and draws a window. It never commands the vehicle.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rov_gui.tools.depth_two_cameras import (  # noqa: E402
    C3_MXID, ITERS, MONO, SCALE, _open, _stats)
from rov_gui.tools.depth_capture_umi import (  # noqa: E402
    LAND_MXID, DEFAULT_DATASET, load_training_rectifier)
from rov_gui.tools.depth_compare import (  # noqa: E402
    Z_NEAR_M, Z_FAR_M, CMAPS, palette_pos, render, side_by_side, near_row)

#: Colormap cycle order for the `c` key.
CMAP_ORDER = ("turbo", "jet", "viridis", "magma", "inferno")


class Pipe:
    """Grab -> rectify -> FoundationStereo -> millimetres, for one camera.

    Holds no Qt and no window: the worker thread owns these, the GUI thread
    only ever reads the last finished frame.
    """

    def __init__(self, name, dev, rectify, to_mm, medium):
        self.name, self.dev = name, dev
        self._rectify, self._to_mm = rectify, to_mm
        self.medium = medium
        self.scale = 1.0
        self.ql = dev.getOutputQueue("left", 2, False)
        self.qr = dev.getOutputQueue("right", 2, False)

    def warm(self, n: int) -> None:
        """Burn frames until auto-exposure settles. The first frames off these
        sensors are saturated white, and a saturated pair has no texture — the
        depth would look real and mean nothing."""
        for _ in range(max(1, n)):
            self.ql.get(), self.qr.get()

    def depth_mm(self, fs):
        l, r = self.ql.get().getCvFrame(), self.qr.get().getCvFrame()
        rl, rr = self._rectify(l, r)
        return self._to_mm(fs.disparity(rl, rr)) * float(self.scale), rl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="depth_live_view", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/depth_live", metavar="DIR",
                    help="where `s` saves snapshots (default: %(default)s)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--domain", choices=("obs", "metric"), default="metric")
    ap.add_argument("--cmap", choices=sorted(CMAPS), default="turbo")
    ap.add_argument("--c3-scale", type=float, default=1.0, metavar="K")
    ap.add_argument("--oak-scale", type=float, default=1.0, metavar="K")
    ap.add_argument("--roi", type=float, default=0.5,
                    help="centre fraction the on-screen numbers use")
    ap.add_argument("--iters", type=int, default=ITERS,
                    help="FoundationStereo GRU iterations (default: the "
                         "training setting). Lower is faster and off-parity.")
    ap.add_argument("--fs-scale", type=float, default=SCALE,
                    help="FoundationStereo input scale (default: the training "
                         "setting). Lower is faster and off-parity.")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--panel-scale", type=int, default=1)
    a = ap.parse_args(argv)

    # ---- DEPTHAI FIRST. See depth_two_cameras: importing cv2 and torch before
    # XLink is initialised makes the USB camera vanish from enumeration for
    # this process only, which reads as a hardware fault.
    import depthai as dai
    print("cameras")
    for k in range(20):
        seen = {d.getMxId(): d.state.name
                for d in dai.Device.getAllAvailableDevices()}
        if LAND_MXID in seen and C3_MXID in seen:
            break
        print(f"  waiting for both to enumerate ({k}): {seen or 'none'}")
        time.sleep(1.5)
    d_umi = _open(LAND_MXID, a.fps)
    print(f"  OAK-D-W open ({LAND_MXID})")
    d_c3 = _open(C3_MXID, a.fps)
    print(f"  C3      open ({C3_MXID})")

    from rov_gui.qt import QtCore, QtGui, QtWidgets, Signal, import_cv2
    cv2 = import_cv2()
    from rov_gui.perception.fstereo import FStereoSession
    from c3_camera.host_depth import StereoRig

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    umi_rect = load_training_rectifier(a.dataset, MONO)
    c3_rig = StereoRig.from_calibration_handler(d_c3.readCalibration(),
                                                mono_size=MONO, alpha=0.0)
    print(f"rigs\n  OAK fx {umi_rect.fx:.4f} baseline {umi_rect.baseline_m * 1000:.3f} mm"
          f"   [calib IN AIR]\n"
          f"  C3  fx {c3_rig.fx_rect:.4f} baseline {c3_rig.baseline_mm:.3f} mm"
          f"   [calib UNDERWATER]")

    def c3_to_mm(disp):
        z = np.zeros(disp.shape, np.float32)
        good = disp > 0.1
        z[good] = c3_rig.fx_rect * c3_rig.baseline_mm / disp[good]
        return z

    pipes = [
        Pipe("OAK-D-W  (land training camera)", d_umi, umi_rect.rectify,
             umi_rect.disparity_to_depth_mm, "IN AIR"),
        Pipe("C3  (deployment camera)", d_c3,
             lambda l, r: (cv2.remap(l, *c3_rig.map_left, interpolation=cv2.INTER_LINEAR),
                           cv2.remap(r, *c3_rig.map_right, interpolation=cv2.INTER_LINEAR)),
             c3_to_mm, "UNDERWATER"),
    ]
    pipes[0].scale, pipes[1].scale = float(a.oak_scale), float(a.c3_scale)

    fs = FStereoSession(iters=int(a.iters), scale=float(a.fs_scale))
    t0 = time.time()
    fs.start_async(on_log=lambda lvl, m: print(f"  [{lvl}] {m}"))
    while not fs.ready and not fs.error and time.time() - t0 < 600:
        time.sleep(0.2)
    if fs.error:
        print(f"FoundationStereo failed: {fs.error}", file=sys.stderr)
        return 3
    print(f"  FoundationStereo ready in {time.time() - t0:.1f} s "
          f"(iters {a.iters}, scale {a.fs_scale}) — ONE session, both cameras")
    print("  warming up the sensors...")
    for p in pipes:
        p.warm(a.warmup)

    # ---------------------------------------------------------------- state
    state = {"domain": a.domain, "cmap": a.cmap, "frozen": False,
             "sheet": None, "hz": 0.0, "stop": False, "shot": 0}
    lock = threading.Lock()

    class Emitter(QtCore.QObject):
        # rov_gui.qt normalises this: PyQt5 spells it pyqtSignal.
        ready = Signal()
    emit = Emitter()

    def compose():
        panels, info = [], []
        for p in pipes:
            z_mm, _ = p.depth_mm(fs)
            zm = np.where(z_mm > 0, z_mm / 1000.0, np.nan)
            st = _stats(z_mm / 1000.0, a.roi)
            st["near_row"] = near_row(zm)
            info.append(st)
            valid = z_mm > 0
            t = palette_pos(np.where(valid, z_mm / 1000.0, Z_FAR_M),
                            state["domain"], Z_NEAR_M, Z_FAR_M)
            sub = (f"calib {p.medium}" if abs(p.scale - 1.0) < 1e-9
                   else f"calib {p.medium} | DEPTH x{p.scale:.3f} APPLIED")
            panels.append(render(t, valid, title=p.name, domain=state["domain"],
                                 cmap=state["cmap"], z_near=Z_NEAR_M,
                                 z_far=Z_FAR_M, subtitle=sub,
                                 scale=a.panel_scale))
        return side_by_side(*panels), info

    def hud(sheet, info, hz):
        u, c = info
        lines = [
            f"domain {state['domain']}   cmap {state['cmap'].upper()}   "
            f"{hz:4.1f} Hz   {'FROZEN' if state['frozen'] else 'live'}",
            f"scale   OAK x{pipes[0].scale:.3f}   C3 x{pipes[1].scale:.3f}"
            f"     [ ] = C3 -/+1%   {{ }} = -/+10%   , . = OAK   1 = 0.752"
            f"   0/r reset",
        ]
        if u.get("n") and c.get("n"):
            ratio = c["p50"] / u["p50"]
            flag = "" if 0.6 < ratio < 1.7 else "   <- framing, not calibration"
            lines += [
                f"centre {100 * a.roi:.0f}%  OAK p50 {u['p50']:6.3f} m   "
                f"C3 p50 {c['p50']:6.3f} m   ratio C3/OAK {ratio:5.3f}{flag}",
                f"near-field row  OAK {u['near_row']:.3f}   C3 {c['near_row']:.3f}"
                f"      (land training = 0.895; 0 = top, 1 = bottom)",
            ]
        else:
            lines.append("no valid pixels in the centre box")
        bar = np.full((22 * len(lines) + 10, sheet.shape[1], 3), 18, np.uint8)
        for i, s in enumerate(lines):
            cv2.putText(bar, s, (10, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (235, 235, 235), 1, cv2.LINE_AA)
        return np.vstack([bar, sheet])

    def worker():
        marks = []
        while not state["stop"]:
            if state["frozen"]:
                time.sleep(0.05)
                continue
            try:
                sheet, info = compose()
            except Exception as e:                               # noqa: BLE001
                print(f"frame failed: {type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(0.2)
                continue
            marks.append(time.monotonic())
            del marks[:-10]
            hz = ((len(marks) - 1) / (marks[-1] - marks[0])
                  if len(marks) > 1 and marks[-1] > marks[0] else 0.0)
            with lock:
                state["sheet"] = hud(sheet, info, hz)
            emit.ready.emit()

    class Window(QtWidgets.QLabel):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("depth — OAK-D-W vs C3, one colour rule")
            self.setAlignment(QtCore.Qt.AlignCenter)
            self.setStyleSheet("background:#121212;")
            self.setMinimumSize(1000, 480)

        def refresh(self):
            with lock:
                img = None if state["sheet"] is None else state["sheet"].copy()
            if img is None:
                return
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            q = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            self.setPixmap(QtGui.QPixmap.fromImage(q.copy()).scaled(
                self.size(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))

        def keyPressEvent(self, e):                              # noqa: N802
            k = e.text()
            key = e.key()
            step = {"[": -0.01, "]": +0.01, "{": -0.10, "}": +0.10}
            if k in step:
                pipes[1].scale = max(0.05, pipes[1].scale * (1 + step[k]))
            elif k in (",", "."):
                pipes[0].scale = max(0.05, pipes[0].scale
                                     * (1 + (-0.01 if k == "," else 0.01)))
            elif k == "1":
                pipes[1].scale = 1.0 / 1.33
            elif k in ("0", "r"):
                pipes[0].scale = pipes[1].scale = 1.0
            elif k == "m":
                state["domain"] = "obs" if state["domain"] == "metric" else "metric"
            elif k == "c":
                i = CMAP_ORDER.index(state["cmap"])
                state["cmap"] = CMAP_ORDER[(i + 1) % len(CMAP_ORDER)]
            elif k == "f":
                state["frozen"] = not state["frozen"]
            elif k == "s":
                with lock:
                    img = None if state["sheet"] is None else state["sheet"].copy()
                if img is not None:
                    state["shot"] += 1
                    p = out / f"live_{state['shot']:03d}.png"
                    cv2.imwrite(str(p), img)
                    print(f"saved {p}   (OAK x{pipes[0].scale:.3f} "
                          f"C3 x{pipes[1].scale:.3f}, {state['domain']}, "
                          f"{state['cmap']})")
            elif k == "q" or key == QtCore.Qt.Key_Escape:
                self.close()

        def closeEvent(self, e):                                 # noqa: N802
            state["stop"] = True
            e.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = Window()
    emit.ready.connect(win.refresh)
    win.resize(1500, 620)
    win.show()
    print("\nwindow open.  [ ] = C3 scale -/+1%   { } = -/+10%   , . = OAK   "
          "1 = 0.752   0/r reset\n"
          "               m = domain   c = colormap   f = freeze   "
          "s = save   q = quit")

    th = threading.Thread(target=worker, name="depth-live", daemon=True)
    th.start()
    try:
        app.exec_()
    finally:
        state["stop"] = True
        th.join(timeout=3.0)
        for d in (d_umi, d_c3):
            try:
                d.close()
            except Exception:                                    # noqa: BLE001
                pass
        try:
            fs.close()
        except Exception:                                        # noqa: BLE001
            pass
    print(f"final scales   OAK x{pipes[0].scale:.4f}   C3 x{pipes[1].scale:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
