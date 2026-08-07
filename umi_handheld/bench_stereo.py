"""bench_stereo.py — measure what a stereo_depth setting actually costs, on the device.

Every number in configs/pipeline*.yaml's comments claims a frame rate or a quality
figure. This is the tool that produces them, so those comments have an artifact behind
them and can be re-run when the settings change.

    python -m umi_handheld.bench_stereo --config configs/pipeline_maxres.yaml
    python -m umi_handheld.bench_stereo --set speckle=off --set median=7x7
    python -m umi_handheld.bench_stereo --sweep median          # the named comparisons

It builds the pipeline through record.py's own build_device_pipeline, so a row here is
the configuration record.py would record with — not a re-implementation that can drift.
(A previous harness that omitted setPostProcessingHardwareResources disagreed with
record.py by 2.3x on a byte-identical filter chain; going through the same builder is
what stops that.)

What it reports, and why these four
-----------------------------------
    fps     device output rate for the depth stream, host doing record.py's polling
    age     median ms a frame has already aged when the host first sees it — the
            preview lag. NOT the frame interval: a device queue that stays full adds
            whole periods to this without moving fps at all, which is how the stereo
            input queues were hiding 95 ms.
    drop%   frames missing, counted by SEQUENCE NUMBER. Wall-clock gaps mislead here —
            trading a few 2-frame gaps for more 1-frame gaps looks worse and is not.
    valid%  share of non-zero depth pixels — density, NOT quality. Optimising it alone
            selects for confident mismatches, which is how the maxres draft ended up
            fastest, densest, and worst.
    tstd    median per-pixel std over time on a STATIC scene (mm). Real depth is steady;
            a mismatch re-rolls every frame, so this is what separates them.
    rough   median |depth step| between horizontally adjacent valid pixels (mm)
    tail    share of adjacent valid pairs disagreeing by more than 100 mm

Rows are comparable only WITHIN one invocation: the camera is pointed at whatever it is
pointed at, and density in particular moves with the scene. Point it at a static, mostly
flat surface and do not touch it while the sweep runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umi_handheld.record import (build_device_pipeline, load_pipeline_config,   # noqa: E402
                                 open_device)
from umi_handheld.camera_model import CameraModel                              # noqa: E402

WARMUP_S = 2.0          # the first frames come out while the device is still settling
QUALITY_FRAMES = 30     # frames kept for tstd; more is slower, 30 measured stable enough


def apply(cfg: dict, sets: dict) -> dict:
    """Apply `--set key=value` shorthands to a loaded config.

    Shorthands rather than dotted paths because these six are the ones worth sweeping,
    and naming them here documents that.
    """
    c = copy.deepcopy(cfg)
    sd = c["record"]["stereo_depth"]
    pp = sd["post_processing"]
    onoff = {"on": True, "off": False, "true": True, "false": False}
    for k, v in sets.items():
        if k in ("speckle", "temporal", "spatial"):
            pp[k]["enable"] = onoff[v.lower()]
        elif k == "threshold":
            pp["threshold"]["enable"] = onoff[v.lower()]
        elif k == "median":
            sd["median_filter"] = v
        elif k == "subpixel":
            sd["subpixel"] = onoff[v.lower()]
        elif k == "extended":
            sd["extended_disparity"] = onoff[v.lower()]
            # max_disparity is not independent — record.py's --dry-run rejects a config
            # where the two disagree, and MinZ is derived from it.
            c["record"]["max_disparity"] = 190 if sd["extended_disparity"] else 95
        elif k == "resolution":
            sd["mono_resolution"] = v
        elif k == "decimation":
            pp["decimation"]["factor"] = int(v)
        elif k == "speckle_range":
            pp["speckle"]["speckle_range"] = int(v)
        elif k == "shaves":
            sd.setdefault("hardware_resources", {})["shaves"] = int(v)
        elif k == "slices":
            sd.setdefault("hardware_resources", {})["slices"] = int(v)
        elif k == "colour":
            c["record"].setdefault("colour", {})["enable"] = onoff[v.lower()]
        else:
            raise SystemExit(f"unknown --set key {k!r}")
    return c


def measure(cfg: dict, seconds: float, src_cam, any_device: bool = False) -> dict:
    pipeline = build_device_pipeline(cfg)
    # Pinned, because the C3 on the lab network answers discovery too and a row
    # measured against the wrong camera looks like a result rather than a mistake.
    with open_device(pipeline, src_cam, any_device) as dev:
        names = dev.getOutputQueueNames()
        q_depth = dev.getOutputQueue("depth", 4, False)
        # The other streams are drained but not measured: leaving them unread would let
        # their queues back up and throttle the device, so the depth rate would be a
        # rate this profile never actually achieves while recording.
        others = [dev.getOutputQueue(n, 4 if n != "color" else 8, False)
                  for n in ("left", "right", "color") if n in names]

        t = time.monotonic()
        while time.monotonic() - t < WARMUP_S:
            q_depth.tryGet()
            for q in others:
                q.tryGet()

        # dai's clock is CLOCK_MONOTONIC; one constant offset puts device timestamps on
        # the same wall clock as time.time(), the same trick record.py:597 uses.
        offset = time.time() - time.monotonic()
        n, kept, t0 = 0, [], time.monotonic()
        ages, seqs = [], []
        while time.monotonic() - t0 < seconds:
            for q in others:
                q.tryGet()
            pkt = q_depth.tryGet()
            if pkt is None:
                continue
            n += 1
            # Age = how stale the frame already is when the host first sees it. This is
            # the number the preview lag is made of, and it is NOT the frame interval:
            # a device queue that stays full adds whole periods to it without changing
            # the rate at all.
            ages.append((time.time() - (pkt.getTimestamp().total_seconds() + offset))
                        * 1000.0)
            seqs.append(int(pkt.getSequenceNum()))
            if len(kept) < QUALITY_FRAMES:
                kept.append(pkt.getFrame().astype(np.float32))
        elapsed = time.monotonic() - t0

    # Drops counted by SEQUENCE NUMBER, not by wall-clock gaps. Watching gaps hides the
    # thing you care about — trading a few 2-frame gaps for more 1-frame gaps looks
    # worse and is not (record.py:294-297).
    drop_pct = None
    if len(seqs) > 1:
        span = seqs[-1] - seqs[0] + 1
        drop_pct = round(100.0 * (span - len(seqs)) / span, 2) if span > 0 else None
    age_ms = round(float(np.median(ages)), 1) if ages else None
    age_p95 = round(float(np.percentile(ages, 95)), 1) if ages else None

    if not kept:
        raise RuntimeError("no depth frames — is the config producing output at all?")
    st = np.stack(kept)
    valid = st > 0
    # tstd only over pixels valid in EVERY frame: a pixel that drops out and returns
    # would otherwise contribute the gap, not the noise.
    always = valid.all(axis=0)
    tstd = float(np.median(st[:, always].std(axis=0))) if always.any() else float("nan")

    last = st[-1]
    a, b = last[:, :-1], last[:, 1:]
    both = (a > 0) & (b > 0)
    step = np.abs(a[both] - b[both])
    nz = st[valid]

    # Depth quantisation, which tstd cannot see: a coarsely quantised value is perfectly
    # steady over time. Disparity is integer unless subpixel is on, so the number of
    # depth values the frame can express is bounded by the disparity level count
    # (190 extended, 95 not, x8 with 3-bit subpixel). `quant_2m` converts that to the
    # step at 2 m — zarr.z_far_m — where it is worst: dz = z^2 / (fx*B*k).
    levels = int(np.unique(last[last > 0]).size)
    band = nz[(nz > 900) & (nz < 1100)]          # a 1 m slice, wide enough to be present
    quant_1m = None
    if band.size > 50:
        u = np.unique(band)
        if u.size > 1:
            quant_1m = round(float(np.median(np.diff(u))), 1)

    return {
        "fps": round(n / elapsed, 2),
        "age_ms": age_ms,
        "age_p95_ms": age_p95,
        "drop_pct": drop_pct,
        "valid_pct": round(100 * float(valid.mean()), 1),
        "tstd_mm": round(tstd, 1),
        "rough_mm": round(float(np.median(step)), 1) if step.size else None,
        "tail_pct": round(100 * float((step > 100).mean()), 1) if step.size else None,
        "closest_mm": round(float(np.percentile(nz, 0.1))) if nz.size else None,
        "levels": levels,
        "quant_1m_mm": quant_1m,
        "depth_size": [last.shape[1], last.shape[0]],
        "frames": n,
    }


# Named comparisons, each answering one question. The point of a sweep over separate
# invocations is that every row sees the same scene.
SWEEPS = {
    # Which post-processing filter actually costs the frame rate at 1280x800?
    "cost": [("as configured", {}),
             ("- colour", {"colour": "off"}),
             ("- temporal", {"temporal": "off"}),
             ("- speckle", {"speckle": "off"}),
             ("- speckle - temporal", {"speckle": "off", "temporal": "off"}),
             ("shaves 8 / slices 6", {"shaves": "8", "slices": "6"})],
    # Can the median filter replace speckle removal? Only where the disparity level
    # count is under the median's 1024 limit — extended x subpixel is 1520 and cannot.
    "median": [("speckle (as configured)", {}),
               ("no filter at all", {"speckle": "off", "temporal": "off"}),
               ("median 3x3", {"speckle": "off", "median": "3x3"}),
               ("median 5x5", {"speckle": "off", "median": "5x5"}),
               ("median 7x7", {"speckle": "off", "median": "7x7"})],
    # What does keeping the near range cost at full resolution?
    "range": [("extended off, subpixel on", {}),
              ("extended off, subpixel on, median 7x7", {"speckle": "off", "median": "7x7"}),
              ("extended ON, subpixel on (no median possible)",
               {"extended": "on", "speckle": "off"}),
              ("extended ON, subpixel OFF, median 7x7",
               {"extended": "on", "subpixel": "off", "speckle": "off", "median": "7x7"})],
    # Extended halves MinZ to 224 mm and doubles the disparity search. Which settings
    # survive that? subpixel and median are mutually exclusive here: extended x subpixel
    # is 190*8 = 1520 levels, over the median filter's 1024 ceiling, so one must go.
    "extended": [
        ("subpixel, no filter (ceiling)",
         {"extended": "on", "speckle": "off", "temporal": "off"}),
        ("subpixel, speckle (naive)", {"extended": "on", "speckle": "on"}),
        ("subpixel, temporal only", {"extended": "on", "speckle": "off"}),
        ("no subpixel, no filter (ceiling)",
         {"extended": "on", "subpixel": "off", "speckle": "off", "temporal": "off"}),
        ("no subpixel, median 3x3",
         {"extended": "on", "subpixel": "off", "speckle": "off", "median": "3x3"}),
        ("no subpixel, median 5x5",
         {"extended": "on", "subpixel": "off", "speckle": "off", "median": "5x5"}),
        ("no subpixel, median 7x7",
         {"extended": "on", "subpixel": "off", "speckle": "off", "median": "7x7"}),
        ("no subpixel, median 7x7, shaves 8/slices 6",
         {"extended": "on", "subpixel": "off", "speckle": "off", "median": "7x7",
          "shaves": "8", "slices": "6"}),
        ("no subpixel, median 7x7, no temporal",
         {"extended": "on", "subpixel": "off", "speckle": "off", "median": "7x7",
          "temporal": "off"}),
        ("no subpixel, median 7x7, 720p (same MinZ, 10% fewer rows)",
         {"extended": "on", "subpixel": "off", "speckle": "off", "median": "7x7",
          "resolution": "720p"}),
    ],
}

HEAD = (f"{'case':40s} {'fps':>6s} {'age':>6s} {'drop%':>6s} {'valid%':>7s} "
        f"{'tstd':>6s} {'rough':>6s} {'tail%':>6s} {'near':>6s} {'lvls':>5s} "
        f"{'q@1m':>6s}  depth")


def _n(v, fmt: str, width: int) -> str:
    return f"{'-':>{width}s}" if v is None else f"{v:{fmt}}"


def row(name: str, r: dict) -> str:
    q = r["quant_1m_mm"]
    return (f"{name:40s} {r['fps']:6.2f} {_n(r['age_ms'], '6.1f', 6)} "
            f"{_n(r['drop_pct'], '6.2f', 6)} {r['valid_pct']:7.1f} "
            f"{r['tstd_mm']:6.1f} {r['rough_mm']:6.1f} {r['tail_pct']:6.1f} "
            f"{r['closest_mm']:6.0f} {r['levels']:5d} "
            f"{(f'{q:.1f}' if q is not None else '-'):>6s}  "
            f"{r['depth_size'][0]}x{r['depth_size'][1]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline_maxres.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one setting; repeatable. Keys: speckle temporal "
                         "spatial threshold median subpixel extended resolution "
                         "decimation speckle_range shaves slices colour")
    ap.add_argument("--sweep", choices=sorted(SWEEPS), help="run a named comparison")
    ap.add_argument("--seconds", type=float, default=7.0, help="measurement window per row")
    ap.add_argument("--json", type=Path, help="also write the results here")
    ap.add_argument("--any-device", action="store_true",
                    help="use whatever DepthAI device is found, not the camera model's")
    a = ap.parse_args()

    base = load_pipeline_config(a.config)
    src_cam = CameraModel.load(base["cameras"]["source"])
    sets = dict(s.split("=", 1) for s in a.set)
    cases = ([(n, {**sets, **o}) for n, o in SWEEPS[a.sweep]] if a.sweep
             else [(", ".join(f"{k}={v}" for k, v in sets.items()) or "as configured", sets)])

    print(f"config  : {base['_path']}")
    print(f"device  : {src_cam.device_mxid or 'any (camera model names none)'}")
    print(f"scene   : keep the camera still and pointed at one surface for all rows\n")
    print(HEAD)
    out = {}
    for name, over in cases:
        r = measure(apply(base, over), a.seconds, src_cam, a.any_device)
        out[name] = r
        print(row(name, r), flush=True)
    if a.json:
        a.json.write_text(json.dumps({"config": base["_path"], "rows": out}, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
