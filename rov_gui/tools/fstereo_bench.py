#!/usr/bin/env python3
"""
fstereo_bench.py — the measurements behind the --fstereo defaults.

Every number quoted in rov_gui/README.md's "--fstereo" section, in the
--fstereo-* help strings and in perception/fstereo.py's constants comes from
one of these sub-commands, and each writes its table to
``rov_gui/tools/fstereo_bench_out/<name>.txt`` so the citation has a path.

    P=~/miniforge3/envs/rovgui-pose/bin/python
    $P -m rov_gui.tools.fstereo_bench link       # CPU only, seconds
    $P -m rov_gui.tools.fstereo_bench sweep      # GPU, ~2 min
    $P -m rov_gui.tools.fstereo_bench graph
    $P -m rov_gui.tools.fstereo_bench jpeg
    $P -m rov_gui.tools.fstereo_bench session
    $P -m rov_gui.tools.fstereo_bench all        # ~8 min, the five above in order

  link     CPU only. How small a REAL 640x400 OV9282 mono frame gets as JPEG
           and what decoding it costs — the case for encoding the pair on the
           device. libjpeg-turbo here vs the Myriad X encoder on the C3, so the
           size ratios are a proxy [예측]; the decode times are host-side real.
  sweep    Network input size x GRU iterations: wall ms, kernel count, and the
           disparity disagreement with the reference setting (iters 16, full
           640x400) on real frames. Shows the forward is launch-bound.
  graph    The shipped session with the CUDA graph on vs off, at each input
           size: capture cost, per-call ms, and bit-equality of the output.
  jpeg     Does a JPEG round-trip on the pair change the disparity? The
           quality side of `link`.
  session  End-to-end infer() through the host rectification + colour-grid
           projection, exactly as the station runs it: ms, fill, provenance.

Frames: the only real 640x400 OV9282 stereo pairs on this machine are the
UMI hand-held recordings under ~/Desktop/data collection/videos (OAK-D-W-97,
H.264, raw unrectified — see oakd_record.py there). The C3's own datasets have
empty left/right directories. Same sensor, different lens, so the absolute
disparities are not the C3's; the COMPARISONS (setting vs setting, JPEG vs
raw) are what these tables are for.

Refuses to run a GPU sub-command while another process holds the GPU: a
number measured against a busy GPU is not a measurement of this code.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")   # before torch/dinov2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np                                       # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "fstereo_bench_out"
VIDEOS = Path("/home/bdml/Desktop/data collection/videos")
EPISODES = ("61", "66", "12")
FRAMES = (10, 90, 170)
MONO_W, MONO_H = 640, 400
POE_MBPS = 90.0

# (w, h, iters, label) — the network input grid the README tables use
SWEEP = [
    (640, 400, 16, "reference"),
    (640, 400, 8, "scale 1.0"),
    (480, 300, 8, "scale 0.75"),
    (320, 200, 16, "scale 0.5, iters 16"),
    (320, 200, 8, "scale 0.5 (shipped)"),
    (320, 200, 4, "scale 0.5"),
    (256, 160, 8, "scale 0.4"),
    (224, 140, 8, "224 wide, isotropic"),
    (224, 140, 4, "224 wide, isotropic"),
    (224, 224, 8, "224x224 anisotropic"),
    (160, 100, 8, "scale 0.25"),
]


# ------------------------------------------------------------------ helpers
class Report:
    """Tee: print now, and keep the lines for the artifact file."""

    def __init__(self, name: str):
        self.name = name
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def save(self, header: list[str]) -> Path:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{self.name}.txt"
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = [f"# fstereo_bench.py {self.name} — {stamp}", *header, ""]
        path.write_text("\n".join(body + self.lines) + "\n")
        print(f"\n-> {path.relative_to(REPO)}")
        return path


def gpu_busy() -> list[str]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    except Exception:                                        # noqa: BLE001
        return []
    # Not ourselves: after the first sub-command has loaded the model, this
    # process IS a compute process, and refusing on that would stop `all`
    # after its first GPU stage (it did).
    me = str(os.getpid())
    return [ln.strip() for ln in r.stdout.splitlines()
            if ln.strip() and ln.split(",")[0].strip() != me]


BUSY_AT_START: list[str] = []
FORCED = False


def refuse_if_busy(force: bool) -> None:
    global BUSY_AT_START, FORCED
    BUSY_AT_START, FORCED = gpu_busy(), bool(force)
    busy = BUSY_AT_START
    if busy and not force:
        sys.exit("REFUSING: the GPU has other compute processes and the "
                 "numbers would be contaminated:\n  " + "\n  ".join(busy)
                 + "\n(--force to run anyway; do not cite the result)")


def gpu_header() -> list[str]:
    import torch
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    return [f"# gpu: {name}, torch {torch.__version__}, other compute "
            f"processes at start: {BUSY_AT_START or 'none'}, at end: "
            f"{gpu_busy() or 'none'}, --force: {FORCED}"]


def load_pairs():
    import cv2
    pairs = []
    for ep in EPISODES:
        capl = cv2.VideoCapture(str(VIDEOS / ep / "left.mp4"))
        capr = cv2.VideoCapture(str(VIDEOS / ep / "right.mp4"))
        for i in FRAMES:
            capl.set(cv2.CAP_PROP_POS_FRAMES, i)
            capr.set(cv2.CAP_PROP_POS_FRAMES, i)
            okl, fl = capl.read()
            okr, fr = capr.read()
            if okl and okr:
                pairs.append((cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
        capl.release()
        capr.release()
    if not pairs:
        sys.exit(f"no frames under {VIDEOS}")
    for L, R in pairs:
        assert L.shape == (MONO_H, MONO_W), L.shape
    return pairs


def session(**kw):
    from rov_gui.perception.fstereo import FStereoSession
    logs: list[str] = []
    s = FStereoSession(**kw)
    s.start_async(on_log=lambda lvl, msg: logs.append(f"[{lvl}] {msg}"))
    while s.loading:
        time.sleep(0.2)
    if s.error:
        sys.exit(s.error)
    return s, logs


def stats(d: np.ndarray) -> str:
    return (f"{np.median(d):6.3f} {np.percentile(d, 90):6.3f} "
            f"{np.percentile(d, 99):6.3f} | {100 * np.mean(d > 1):4.1f}% "
            f"{100 * np.mean(d > 2):4.1f}%")


# ------------------------------------------------------------------- link
def cmd_link(a) -> None:
    import cv2
    rep = Report("link")
    pairs = load_pairs()
    frames = [L for L, _ in pairs]
    raw = MONO_W * MONO_H
    rep(f"{len(frames)} real {MONO_W}x{MONO_H} mono frames; raw = {raw / 1e3:.0f} kB/frame")
    rep(f"{'q':>3} | {'kB':>6} {'ratio':>6} | {'pair Mbit/s @15':>15} {'@20':>6} | "
        f"{'decode ms':>9} | {'PSNR dB':>7}")
    for q in (70, 80, 90, 95):
        sizes, dec, psnr = [], [], []
        for g in frames:
            ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])
            sizes.append(len(buf))
            t = time.perf_counter()
            d = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            dec.append((time.perf_counter() - t) * 1e3)
            mse = np.mean((d.astype(np.float32) - g.astype(np.float32)) ** 2)
            psnr.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
        kb = np.median(sizes) / 1e3
        rep(f"{q:3} | {kb:6.1f} {raw / np.median(sizes):5.1f}x | "
            f"{2 * np.median(sizes) * 8 * 15 / 1e6:15.1f} "
            f"{2 * np.median(sizes) * 8 * 20 / 1e6:6.1f} | "
            f"{np.median(dec):9.2f} | {np.median(psnr):7.1f}")
    rep(f"raw pair: {2 * raw * 8 * 15 / 1e6:.1f} Mbit/s @15, {2 * raw * 8 * 20 / 1e6:.1f} @20; "
        f"one raw frame on a {POE_MBPS:.0f} Mbit/s link = {raw * 8 / (POE_MBPS * 1e6) * 1e3:.1f} ms wire time")
    rep.save([f"# frames: {VIDEOS} episodes {EPISODES} frames {FRAMES} (left)",
              "# encoder: OpenCV libjpeg-turbo on the host — a PROXY for the C3's "
              "Myriad X JPEG encoder; ratios are [예측], decode ms are real"])


# ------------------------------------------------------------------ sweep
def _disp_at(s, torch, cv2, L, R, w, h, iters):
    """Disparity at 640x400, computed at w x h — through the session's own
    forward (graph or eager, whatever it is set to)."""
    H, W = L.shape
    l = cv2.resize(L, (w, h), interpolation=cv2.INTER_AREA)
    r = cv2.resize(R, (w, h), interpolation=cv2.INTER_AREA)

    def tt(img):
        rgb = np.repeat(img[:, :, None], 3, axis=2)
        return torch.as_tensor(rgb).cuda().float()[None].permute(0, 3, 1, 2)

    tl, tr = tt(l), tt(r)
    padder = s._padder(tl.shape, divis_by=32, force_square=False)
    tl, tr = padder.pad(tl, tr)
    s.iters = iters
    d = s._forward(tl, tr)
    d = padder.unpad(d.float()).cpu().numpy().reshape(h, w)
    torch.cuda.synchronize()
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) * (W / w), tuple(tl.shape[-2:])


def _kernel_count(s, torch, cv2, L, R, w, h, iters) -> int:
    from torch.profiler import ProfilerActivity, profile
    _disp_at(s, torch, cv2, L, R, w, h, iters)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _disp_at(s, torch, cv2, L, R, w, h, iters)
    n = 0
    for e in prof.key_averages():
        dt = getattr(e, "device_type", None)
        if dt is not None and "cuda" in str(dt).lower():
            n += e.count
    return n


def cmd_sweep(a) -> None:
    refuse_if_busy(a.force)
    import cv2
    rep = Report("sweep")
    pairs = load_pairs()
    s, _ = session(iters=16, scale=1.0, graph=False)      # EAGER: kernel counts
    torch = s._torch
    rep(f"{len(pairs)} real pairs; eager forward; disagreement vs reference "
        f"(iters 16, {MONO_W}x{MONO_H}) in {MONO_W}-px units, 20-px border excluded")
    ref = [_disp_at(s, torch, cv2, L, R, MONO_W, MONO_H, 16)[0] for L, R in pairs]
    rep(f"{'input':>9} {'padded':>9} {'it':>2} | {'wall ms':>7} {'kernels':>7} | "
        f"{'med|d|':>6} {'p90':>6} {'p99':>6} | {'>1px':>5} {'>2px':>5} | note")
    for w, h, iters, label in SWEEP:
        for _ in range(2):
            _, padded = _disp_at(s, torch, cv2, *pairs[0], w, h, iters)
        ts, dd = [], []
        for k, (L, R) in enumerate(pairs):
            t = time.perf_counter()
            d, _ = _disp_at(s, torch, cv2, L, R, w, h, iters)
            ts.append((time.perf_counter() - t) * 1e3)
            dd.append(np.abs(d - ref[k])[20:-20, 20:-20].ravel())
        nk = _kernel_count(s, torch, cv2, *pairs[0], w, h, iters)
        rep(f"{w:4}x{h:<4} {padded[1]:4}x{padded[0]:<4} {iters:2} | {np.median(ts):7.1f} "
            f"{nk:7} | {stats(np.concatenate(dd))} | {label}")
    s.close()
    rep.save(gpu_header() + [f"# frames: {VIDEOS} episodes {EPISODES} frames {FRAMES}"])


# ------------------------------------------------------------------ graph
def cmd_graph(a) -> None:
    refuse_if_busy(a.force)
    import cv2
    rep = Report("graph")
    pairs = load_pairs()
    s, logs = session(iters=8, scale=0.5, graph=True)
    torch = s._torch
    rep(f"{len(pairs)} real pairs; the shipped FStereoSession, graph on vs off, "
        f"per input size; ms = one _disp_at call (cv2 resize + H2D + forward + "
        f"D2H + resize-back), median over the pairs")
    rep(f"{'input':>9} {'it':>2} | {'capture ms':>10} {'replay ms':>9} {'eager ms':>8} | "
        f"{'max|replay-eager| px':>20} {'equal':>5} | note")
    for w, h, iters, label in SWEEP:
        if (w, h, iters) == (MONO_W, MONO_H, 16):
            continue                       # the reference setting is not a candidate
        s.graph, s._graph, s._graph_error = True, None, ""
        n0 = len(logs)
        t = time.perf_counter()
        _disp_at(s, torch, cv2, *pairs[0], w, h, iters)     # captures
        cap_ms = (time.perf_counter() - t) * 1e3
        if not s.graph:
            rep(f"{w:4}x{h:<4} {iters:2} | CAPTURE FAILED: {s._graph_error[:90]}")
            continue
        rep_ts, rep_out = [], []
        for L, R in pairs:
            t = time.perf_counter()
            d, _ = _disp_at(s, torch, cv2, L, R, w, h, iters)
            rep_ts.append((time.perf_counter() - t) * 1e3)
            rep_out.append(d)
        s.graph = False
        _disp_at(s, torch, cv2, *pairs[0], w, h, iters)
        eag_ts, diff, equal = [], 0.0, True
        for k, (L, R) in enumerate(pairs):
            t = time.perf_counter()
            d, _ = _disp_at(s, torch, cv2, L, R, w, h, iters)
            eag_ts.append((time.perf_counter() - t) * 1e3)
            diff = max(diff, float(np.abs(d - rep_out[k]).max()))
            equal = equal and bool(np.array_equal(d, rep_out[k]))
        rep(f"{w:4}x{h:<4} {iters:2} | {cap_ms:10.0f} {np.median(rep_ts):9.1f} "
            f"{np.median(eag_ts):8.1f} | {diff:20.4f} {str(equal):>5} | {label}")
    s.close()
    rep.save(gpu_header() + [f"# frames: {VIDEOS} episodes {EPISODES} frames {FRAMES}",
                             "# capture ms includes three eager warm-up passes"])


# ------------------------------------------------------------------- jpeg
def cmd_jpeg(a) -> None:
    refuse_if_busy(a.force)
    import cv2
    rep = Report("jpeg")
    pairs = load_pairs()
    s, _ = session(iters=8, scale=0.5, graph=True)
    s.disparity(*pairs[0])
    base = [s.disparity(L, R) for L, R in pairs]

    def jpeg(img, q):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    rep(f"{len(pairs)} real pairs; shipped setting (iters 8, scale 0.5, graph); "
        f"disparity after a JPEG round-trip of BOTH images vs the raw-input disparity, "
        f"{MONO_W}-px units")
    rep(f"{'q':>3} | {'med|d|':>6} {'p90':>6} {'p99':>6} | {'>1px':>5} {'>2px':>5}")
    for q in (80, 90, 95):
        dd = []
        for k, (L, R) in enumerate(pairs):
            d = s.disparity(jpeg(L, q), jpeg(R, q))
            dd.append(np.abs(d - base[k])[20:-20, 20:-20].ravel())
        rep(f"{q:3} | {stats(np.concatenate(dd))}")
    rep("for scale: the shipped setting's own disagreement with the reference "
        "setting is med 0.165 px, >1 px on 11.1% (see sweep.txt)")
    s.close()
    rep.save(gpu_header() + [f"# frames: {VIDEOS} episodes {EPISODES} frames {FRAMES}",
                             "# JPEG: OpenCV libjpeg-turbo, a proxy for the device encoder"])


# ---------------------------------------------------------------- session
def cmd_session(a) -> None:
    refuse_if_busy(a.force)
    from c3_camera.host_depth import bench_rig
    rep = Report("session")
    pairs = load_pairs()
    rig = bench_rig((MONO_W, MONO_H), with_color=True, color_size=(400, 250))
    rep(f"{len(pairs)} real pairs through infer(): host rectify -> network -> mm -> "
        f"400x250 colour grid -> NEAREST to 640x360 (the station's path)")
    rep(f"{'setting':>26} | {'load s':>6} {'1st ms':>6} {'infer ms':>8} {'Hz':>5} | "
        f"{'native %':>8} {'fill %':>6} {'loss %':>6} | provenance")
    outs = {}
    for label, kw in (("graph, scale 0.5, iters 8", dict(iters=8, scale=0.5, graph=True)),
                      ("eager, scale 0.5, iters 8", dict(iters=8, scale=0.5, graph=False)),
                      ("graph, 224x224, iters 8", dict(iters=8, size=(224, 224), graph=True)),
                      ("graph, 224x140, iters 4", dict(iters=4, size=(224, 140), graph=True))):
        s, logs = session(**kw)
        t = time.perf_counter()
        s.infer(*pairs[0], rig, out_size=(640, 360))
        first = (time.perf_counter() - t) * 1e3
        ts, fills, natives, maps = [], [], [], []
        for L, R in pairs:
            t = time.perf_counter()
            o = s.infer(L, R, rig, out_size=(640, 360))
            ts.append((time.perf_counter() - t) * 1e3)
            fills.append(o["valid_out"])
            natives.append(o["valid_native"])
            maps.append(o["depth_mm"])
        d = s.describe()
        load_s = s.load_seconds            # BEFORE close(): it reads 0.0 after
        s.close()
        outs[label] = maps
        # native = pixels the network gave a positive disparity (no Z cut);
        # loss = what the forward scatter onto the colour grid then drops.
        rep(f"{label:>26} | {load_s:6.1f} {first:6.0f} {np.median(ts):8.1f} {1000 / np.median(ts):5.1f} | "
            f"{np.median(natives):8.2f} {np.median(fills):6.2f} "
            f"{np.median(np.array(natives) - np.array(fills)):6.2f} | "
            f"cuda_graph={d['cuda_graph']} input={d['cuda_graph_input']} "
            f"err={d['cuda_graph_error']}")
        for m in logs:
            if "graph" in m:
                rep(f"{'':>26}   {m}")
    g, e = outs["graph, scale 0.5, iters 8"], outs["eager, scale 0.5, iters 8"]
    diff = max(float(np.abs(x.astype(np.int32) - y.astype(np.int32)).max()) for x, y in zip(g, e))
    rep(f"graph vs eager depth maps (scale 0.5, iters 8), {len(g)} frames: max |d| = {diff:.0f} mm")
    rep.save(gpu_header() + [f"# frames: {VIDEOS} episodes {EPISODES} frames {FRAMES}",
                             "# rig: c3_camera.host_depth.bench_rig (synthetic C3-like geometry)"])


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("link", "sweep", "graph", "jpeg", "session", "all"))
    p.add_argument("--force", action="store_true",
                   help="run even if another process holds the GPU (do not cite)")
    a = p.parse_args(argv)
    cmds = {"link": cmd_link, "sweep": cmd_sweep, "graph": cmd_graph,
            "jpeg": cmd_jpeg, "session": cmd_session}
    order = ("link", "sweep", "graph", "jpeg", "session") if a.what == "all" else (a.what,)
    for name in order:
        print(f"\n===== {name} =====")
        cmds[name](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
