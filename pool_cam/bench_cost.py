#!/usr/bin/env python3
"""What the pool cameras cost the machine, measured: CPU of the camera process
and its ffmpeg encoders, and of the station's side (this process, standing in
for rov_gui: the same PoolCamClient receiving the same previews).

    python pool_cam/bench_cost.py                       # every pool camera found
    python pool_cam/bench_cost.py --device NE=/dev/v4l/by-path/...  --seconds 30
    python pool_cam/bench_cost.py --fake 4 --size 640x360           # no hardware

Phases, each --seconds long: idle (streaming, tab hidden), preview (tab on
screen, one tile per camera), record (tab on screen + recording). CPU is
summed utime+stime from /proc over the phase, in cores (1.00 = one core busy).
The recording lands in a temp folder and is deleted unless --keep.

Prints a report; --out FILE also writes it (cite that file, not the console).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rov_gui.poolcam import POOL_CAM_PY, PoolCamClient   # noqa: E402

TICK = os.sysconf("SC_CLK_TCK")


def cpu_s(pid):
    """utime+stime of pid and every descendant, in seconds."""
    total, todo = 0.0, [pid]
    while todo:
        p = todo.pop()
        try:
            f = Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()
            total += (int(f[11]) + int(f[12])) / TICK
            for t in Path(f"/proc/{p}/task").iterdir():
                kids = (t / "children").read_text().split()
                todo += [int(k) for k in kids]
        except (OSError, IndexError, ValueError):
            pass
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", action="append", default=[])
    ap.add_argument("--fake", type=int, default=0)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", default="30")
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--tile", default="960x540", help="preview size per camera")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    from rov_gui.qt import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841

    argv = [sys.executable, str(POOL_CAM_PY), "--serve", "--size", a.size,
            "--fps", a.fps, "--encoder", a.encoder]
    argv += ["--fake", str(a.fake)] if a.fake else sum((["--device", d] for d in a.device), [])
    logs = []
    client = PoolCamClient(argv, log=lambda lvl, m: logs.append(f"[{lvl}] {m}"))
    out = Path(tempfile.mkdtemp(prefix="pool_bench_"))
    lines = [f"# pool_cam/bench_cost.py  {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"# argv: {' '.join(argv[1:])}", f"# host cores: {os.cpu_count()}"]
    try:
        client.start()
        end = time.monotonic() + 20
        while client.snapshot()[0] is None and time.monotonic() < end:
            time.sleep(0.1)
        hello = client.snapshot()[0]
        if hello is None:
            sys.exit("the camera process never said hello:\n" + "\n".join(logs))
        labels = [c["label"] for c in hello["cams"]]
        lines.append(f"# cameras: {labels}  codec: {hello.get('codec')}")
        time.sleep(3)                                    # let them settle
        tw, th = (int(v) for v in a.tile.split("x"))
        seen = {lbl: -1 for lbl in labels}

        def phase(name):
            child = client._proc.pid
            c0, p0, t0 = cpu_s(child), time.process_time(), time.monotonic()
            frames = 0
            while time.monotonic() - t0 < a.seconds:
                for lbl in labels:                       # what the GUI tick does
                    got = client.take_frame(lbl, seen[lbl])
                    if got is not None:
                        seen[lbl] = got[0]
                        frames += 1
                time.sleep(1 / 60)
            dt = time.monotonic() - t0
            st = client.snapshot()[1] or {}
            fps = ", ".join(f"{c['label']} {c['fps']:.1f}" for c in st.get("cams", []))
            lines.append(f"{name:8s} camera-process+ffmpeg {(cpu_s(child) - c0) / dt:5.2f} cores"
                         f"   station side {(time.process_time() - p0) / dt:5.2f} cores"
                         f"   previews {frames / dt:5.1f}/s   camera fps: {fps}")

        client.set_preview({})
        phase("idle")
        client.set_preview({lbl: (tw, th) for lbl in labels})
        phase("preview")
        client.start_recording(time.monotonic(), out / "pool_bench")
        phase("record")
        client.stop_recording()
    finally:
        client.stop()
    keys = ("encoder", "frames", "seconds", "camera_frames", "repeated", "skipped",
            "dropped", "complete", "error")
    for f in sorted(f for f in (out / "pool_bench").glob("*.json")
                    if f.name != "session.json"):
        meta = json.loads(f.read_text())
        lines.append(f"# {f.name}: " + "  ".join(f"{k}={meta.get(k)}" for k in keys))
    lines += [f"# log: {m}" for m in logs if "error" in m or "warn" in m]
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(report)
    if not a.keep:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
