#!/usr/bin/env python3
"""fstereo_proc_parity.py — does ``--fstereo-process`` produce the SAME depth?

The same recorded mono pairs go through the in-station ``FStereoSession`` and
through ``RemoteFStereoSession`` (the real network in a child process, as
``--fstereo-process`` runs it), one after the other on an idle GPU, and the
outputs the station consumes are compared: ``depth_native`` (what the policy
gets) and ``depth_mm`` (panel / FoundationPose), plus the valid percentages.
Also measures the real child's load time, first frame (graph capture) and how
long ``close()`` takes (it must fit the worker's 4000 ms stop budget).

    P=~/miniforge3/envs/rovgui-pose/bin/python
    $P rov_gui/tools/fstereo_proc_parity.py            # scale 0.5 and 0.75, iters 8
    $P rov_gui/tools/fstereo_proc_parity.py --scales 0.5 --repeat 3

Inputs: the 9 land OAK-D-W pairs of rov_gui/tools/fstereo_bench.load_pairs()
with c3_camera.host_depth.bench_rig (alpha 0.5) — no C3 pairs are on disk.
Writes rov_gui/tools/fstereo_bench_out/proc_parity_<stamp>.json. Refuses a busy
GPU (fstereo_bench.refuse_if_busy). Opens no camera, no vehicle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                             # noqa: E402

OUT_DIR = REPO / "rov_gui" / "tools" / "fstereo_bench_out"
OUT_SIZE = (640, 360)


def _wait(s, limit=180.0):
    t0 = time.monotonic()
    while s.loading and time.monotonic() - t0 < limit:
        time.sleep(0.02)
    if s.error or not s.ready:
        raise SystemExit(f"session did not load: {s.error or 'not ready'}")


def _run(session, pairs, rig, repeat):
    outs, ms = [], []
    for _ in range(repeat):
        for left, right in pairs:
            t = time.perf_counter()
            o = session.infer(left, right, rig, out_size=OUT_SIZE)
            ms.append((time.perf_counter() - t) * 1e3)
            outs.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                         for k, v in o.items() if k != "rect_left"})
    return outs, ms


def _cmp(a, b):
    res = {"n": len(a), "native_equal": 0, "mm_equal": 0,
           "native_max_abs_mm": 0, "mm_max_abs_mm": 0,
           "native_diff_px_pct_max": 0.0, "valid_equal": 0}
    for x, y in zip(a, b):
        dn = np.abs(x["depth_native"].astype(np.int32)
                    - y["depth_native"].astype(np.int32))
        dm = np.abs(x["depth_mm"].astype(np.int32) - y["depth_mm"].astype(np.int32))
        res["native_equal"] += int(np.array_equal(x["depth_native"], y["depth_native"]))
        res["mm_equal"] += int(np.array_equal(x["depth_mm"], y["depth_mm"]))
        res["native_max_abs_mm"] = max(res["native_max_abs_mm"], int(dn.max()))
        res["mm_max_abs_mm"] = max(res["mm_max_abs_mm"], int(dm.max()))
        res["native_diff_px_pct_max"] = max(res["native_diff_px_pct_max"],
                                            round(100.0 * float(np.mean(dn > 0)), 4))
        res["valid_equal"] += int(all(x[k] == y[k] for k in
                                      ("valid_native", "valid_out", "filled_out")))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", default="0.5,0.75")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    from c3_camera.host_depth import bench_rig
    from rov_gui.perception.fstereo import FStereoSession
    from rov_gui.perception.fstereo_proc import RemoteFStereoSession
    from rov_gui.tools import fstereo_bench as fb

    fb.refuse_if_busy(a.force)
    pairs = fb.load_pairs()
    rig = bench_rig((fb.MONO_W, fb.MONO_H), with_color=True,
                    color_size=(400, 250), alpha=0.5)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = {"tool": "rov_gui/tools/fstereo_proc_parity.py", "when": stamp,
           "pairs": len(pairs), "repeat": a.repeat, "iters": a.iters,
           "out_size": list(OUT_SIZE), "rig": "bench_rig alpha 0.5, align 400x250",
           "gpu": fb.gpu_header(), "results": []}
    for scale in [float(x) for x in a.scales.split(",") if x]:
        print(f"== scale {scale}: in-station", flush=True)
        s = FStereoSession(iters=a.iters, scale=scale, graph=True)
        t0 = time.monotonic()
        s.start_async(on_log=lambda lvl, m: print(f"  [{lvl}] {m}", flush=True))
        _wait(s)
        load_in = time.monotonic() - t0
        ins, ms_in = _run(s, pairs, rig, a.repeat)
        t0 = time.monotonic()
        s.close()
        close_in = time.monotonic() - t0
        del s

        print(f"== scale {scale}: child process", flush=True)
        r = RemoteFStereoSession(iters=a.iters, scale=scale, graph=True)
        t0 = time.monotonic()
        r.start_async(on_log=lambda lvl, m: print(f"  [child {lvl}] {m}", flush=True))
        _wait(r)
        load_child = time.monotonic() - t0
        outs, ms_ch = _run(r, pairs, rig, a.repeat)
        meta = r.process_meta()
        desc = r.describe()
        t0 = time.monotonic()
        r.close()
        close_child = time.monotonic() - t0

        cmp = _cmp(ins, outs)
        q = lambda v: {"p50": round(float(np.median(v[1:])), 2),
                       "max": round(float(np.max(v[1:])), 2)} if len(v) > 1 else None
        res = {"scale": scale, "compare": cmp,
               "in_station": {"load_s": round(load_in, 2),
                              "first_frame_ms": round(ms_in[0], 1),
                              "infer_ms_after_first": q(ms_in),
                              "close_s": round(close_in, 2)},
               "child": {"load_s_station": round(load_child, 2),
                         "first_frame_ms": round(ms_ch[0], 1),
                         "infer_ms_after_first": q(ms_ch),
                         "close_s": round(close_child, 2),
                         "exit_code": r.process_meta()["exit_code"],
                         "process_meta": meta,
                         "cuda_graph": desc.get("cuda_graph"),
                         "ckpt_sha1_first_8mib": desc.get("ckpt_sha1_first_8mib")}}
        out["results"].append(res)
        print(json.dumps(res["compare"]), flush=True)
        print(f"  close: in-station {close_in:.2f} s, child {close_child:.2f} s "
              f"(exit {res['child']['exit_code']})", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"proc_parity_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"-> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
