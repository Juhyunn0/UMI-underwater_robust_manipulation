#!/usr/bin/env python3
"""make_depth_trajectory_video.py — camera | FoundationStereo depth | trajectory, in one mp4.

    python data_collection/make_depth_trajectory_video.py "/home/bdml/Desktop/data collection"

Writes `<dataset>/slam/<ep>/trajectory_depth.mp4`, a 1920x480 three-panel video:

    [ what the camera saw ] [ the depth the policy was trained on ] [ the SLAM trajectory ]

The outer two panels are the existing `trajectory.mp4` (from UMI_Underwater's
`visualize_trajectory.py`) split down the middle; this tool inserts a depth panel between
them and re-muxes. It does NOT re-render the trajectory, so the picture on the right is the
same picture that was already reviewed -- and an episode whose `trajectory.mp4` is missing is
skipped with a pointer at the script that makes it, rather than being drawn differently.

Which depth
-----------
`--depth-source fs` (default) is the direct FoundationStereo product,
`<dataset>/depth/<ep>/depth.zarr` (float16 mm, 0 = invalid, on the RECTIFIED-LEFT grid at
640x400, fx_rect 249.5656 px) -- written by `UMI_Underwater/replay_foundation_stereo.py`.
That is the whole frame at full resolution, which is what you want next to a full-frame
camera view.

`--depth-source obs` is the 224x224 uint8 tensor the diffusion policy actually consumed,
`<dataset>/dataset_depth.zarr` `data/camera0_depth` -- i.e. the fs depth after warp to the
C3's optics, centre crop, resize and inverse-depth normalisation
(`umi_handheld/build_dp_depth_zarr.py`). Use it to see the policy's own field of view.

Both are one row per source video frame for this dataset (every episode's
`kept_frame_range` covers the whole episode), and the mapping is taken from the artifacts
themselves -- `frame_index` for `fs`, `dataset_plan[...]['episode']` + `episode_ends` for
`obs` -- never from position in a list. Positional joins are exactly how ep28's deletion
shifts every later label by one.

Colour
------
The colour rule is imported from `rov_gui/tools/depth_compare.py`, unchanged: fixed
endpoints out of the training store's own `.zattrs['obs']` (0.20-3.00 m), no per-frame
auto-range, no histogram equalisation, warm = NEAR, invalid = BLACK and distinguishable
from z_far. It is burned into every frame and also written to
`trajectory_depth.colour_rule.json` beside the video. Four renderers in this project used to
disagree about all of that; laying two of them side by side measured the renderers rather
than the depth, so this file gets its ramp from the one module that owns it.

Run it in the env that produced the depth (numpy, cv2, zarr, av):

    ~/miniforge3/envs/fstereo/bin/python data_collection/make_depth_trajectory_video.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rov_gui.tools.depth_compare import (          # noqa: E402
    CMAPS, Z_FAR_M, Z_NEAR_M, colorize, colour_rule_json, mm_to_t_valid,
    obs_to_metres, obs_to_t_valid, palette_pos)

# The panel geometry and the drawing helpers live in rov_gui/depth_panel.py so
# the station's run video draws the identical panel; PANEL_W/PANEL_H are still
# this file's own layout unit (visualize_trajectory.py's panel is the same size).


from rov_gui.depth_panel import (BAR_H, BG, HEADER_H, IMAGE_H,  # noqa: E402
                                 PANEL_H, PANEL_W, colour_bar_h, depth_panel,
                                 fit_into, missing_panel, rule_line, text)

# -------------------------------------------------------------------- depth sources

class FsDepth:
    """`depth/<ep>/depth.zarr` -- FoundationStereo's own output, float16 mm."""

    kind = "fs"

    def __init__(self, dataset: Path, ep: int, domain: str):
        self.path = dataset / "depth" / str(ep) / "depth.zarr"
        if not self.path.exists():
            raise FileNotFoundError(
                f"no FoundationStereo depth for ep {ep}: {self.path}\n"
                f"  make it with UMI_Underwater/replay_foundation_stereo.py "
                f"<dataset> --episodes {ep} --fs-repo <FoundationStereo>")
        z = zarr.open(str(self.path), mode="r")
        self.depth = z["depth_mm"]
        self.frame_index = np.asarray(z["frame_index"]).astype(int)
        self.attrs = dict(z.attrs)
        self.domain = domain
        # source frame -> row. Built from the store's own frame_index so a strided
        # replay (stride > 1) holds the last depth instead of sliding out of sync.
        self._row = {int(f): r for r, f in enumerate(self.frame_index)}
        self.z_near, self.z_far = Z_NEAR_M, Z_FAR_M
        self.n_source = int(self.frame_index.max()) + 1 if len(self.frame_index) else 0

    def header(self, ep, i, n):
        a = self.attrs
        ck = Path(str(a.get("checkpoint", "?")))
        return [
            f"FoundationStereo depth   depth/{ep}/depth.zarr   "
            f"{self.depth.shape[2]}x{self.depth.shape[1]} mm, 0 = invalid"
            f"   ep {ep}  frame {i}{f'/{n - 1}' if n else ''}",
            f"{ck.parent.name}/{ck.name}  iters {a.get('iters', '?')}  "
            f"scale {a.get('scale', '?')}  "
            f"fx_rect {float(a.get('fx_rect_px', float('nan'))):.2f} px  "
            f"({a.get('frame_of', '?')} grid)",
        ]

    def row_for(self, i):
        """(row, note) for source video frame i, or (None, message).

        The one place the frame -> row question is answered, so a hover readout and
        the pixel it is hovering over cannot come from different rows.
        """
        row = self._row.get(i)
        if row is not None:
            return row, ""
        earlier = [f for f in self._row if f < i]
        if not earlier:
            return None, "no depth for this frame"
        held = max(earlier)
        return self._row[held], (f"held from frame {held} "
                                 f"(stride {self.attrs.get('stride', '?')})")

    def frame(self, i):
        """(t, valid, note) for source video frame i, or (None, None, message)."""
        row, note = self.row_for(i)
        if row is None:
            return None, None, note
        d = np.asarray(self.depth[row]).astype(np.float32)
        t, valid = mm_to_t_valid(d, self.domain, self.z_near, self.z_far)
        return t, valid, note

    def metres(self, i):
        """(depth in METRES with NaN where unmeasured, row, note), or (None, None, msg).

        The number behind the colour, for readers that quote a distance rather than
        paint one (data_collection/make_depth_trajectory_html.py). `row` is returned so a
        caller can cache per stored row instead of per video frame.
        """
        row, note = self.row_for(i)
        if row is None:
            return None, None, note
        d = np.asarray(self.depth[row]).astype(np.float32)
        return np.where(d > 0, d / 1000.0, np.nan), row, note


class ObsDepth:
    """`dataset_depth.zarr` -- the 224x224 tensor the policy was trained on."""

    kind = "obs"

    def __init__(self, dataset: Path, store_path: Path | None, domain: str):
        p = store_path or (dataset / "dataset_depth.zarr")
        if not p.exists() and (dataset / "dataset_depth.zarr.zip").exists():
            p = dataset / "dataset_depth.zarr.zip"
        if not p.exists():
            raise FileNotFoundError(f"training depth store not found: {p}")
        self.path = p
        if p.suffix == ".zip":
            self.root = zarr.open(zarr.storage.ZipStore(str(p), mode="r"), mode="r")
        else:
            self.root = zarr.open(str(p), mode="r")
        self.arr = self.root["data/camera0_depth"]
        self.ends = np.asarray(self.root["meta/episode_ends"]).astype(int)
        self.attrs = dict(self.root.attrs)
        obs = self.attrs.get("obs", {})
        self.z_near = float(obs.get("z_near_m", Z_NEAR_M))
        self.z_far = float(obs.get("z_far_m", Z_FAR_M))
        self.domain = domain
        self.plan = self.attrs.get("dataset_plan", [])
        if len(self.plan) != len(self.ends):
            raise ValueError(f"{p}: dataset_plan {len(self.plan)} != "
                             f"episode_ends {len(self.ends)}")

    def for_episode(self, ep: int):
        """Locate ep BY ITS 'episode' FIELD, not by position -- ep28 is deleted, so the
        k-th plan entry is not the k-th source episode from there on."""
        hits = [k for k, e in enumerate(self.plan) if int(e["episode"]) == ep]
        if not hits:
            raise KeyError(f"episode {ep} is not in {self.path.name}'s dataset_plan "
                           f"(it was dropped from the training set)")
        k = hits[0]
        lo, hi = (int(v) for v in self.plan[k]["kept_frame_range"])
        start = 0 if k == 0 else int(self.ends[k - 1])
        end = int(self.ends[k])
        if end - start != hi - lo:
            raise ValueError(f"ep {ep}: store slice {end - start} frames != "
                             f"kept_frame_range {hi - lo}")
        self._ep, self._lo, self._hi, self._start = ep, lo, hi, start
        self.n_source = hi
        return self

    def header(self, ep, i, n):
        return [
            f"training obs   {self.path.name}  data/camera0_depth"
            f"[{self._start}+{self._hi - self._lo}]  224x224 ch0 = inverse depth"
            f"   ep {ep}  frame {i}{f'/{n - 1}' if n else ''}",
            f"warp to C3 optics -> centre crop -> 224 -> normalise "
            f"({self.z_near:.2f}-{self.z_far:.2f} m);  kept frames "
            f"{self._lo}-{self._hi - 1} of ep {ep}",
        ]

    def row_for(self, i):
        if not (self._lo <= i < self._hi):
            return None, "outside the kept span (not in the training set)"
        return self._start + (i - self._lo), ""

    def metres(self, i):
        """See FsDepth.metres. Quantisation is coarse near z_far here -- one u8 step is
        ~200 mm at 2.8 m -- because that is what the policy was trained on."""
        row, note = self.row_for(i)
        if row is None:
            return None, None, note
        return obs_to_metres(np.asarray(self.arr[row]), self.z_near, self.z_far), row, note

    def frame(self, i):
        row, note = self.row_for(i)
        if row is None:
            return None, None, note
        obs = np.asarray(self.arr[row])
        if self.domain == "obs":
            t, valid = obs_to_t_valid(obs)
        else:
            z = obs_to_metres(obs, self.z_near, self.z_far)
            valid = np.isfinite(z)
            t = palette_pos(np.where(valid, z, self.z_far), "metric",
                            self.z_near, self.z_far)
        return t, valid, ""


# ------------------------------------------------------------------------------ video

def decode(path):
    """Yield BGR frames of an mp4."""
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="bgr24")


def probe(path):
    with av.open(str(path)) as container:
        s = container.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else 30.0
        return s.codec_context.width, s.codec_context.height, fps, int(s.frames or 0)


def letterbox(img):
    return fit_into(img, PANEL_W, PANEL_H, interp=cv2.INTER_AREA)


def render_episode(dataset: Path, slam_root: Path, ep: int, src, out: Path, *,
                   camera: str, domain: str, cmap: str, crf: int) -> Path:
    traj_path = slam_root / str(ep) / "trajectory.mp4"
    tw, th, fps, n_frames = probe(traj_path)
    split = (tw == 2 * PANEL_W)
    if camera == "split" and not split:
        raise ValueError(f"ep {ep}: {traj_path.name} is {tw}x{th}, not two panels; "
                         f"use --camera rgb or --camera left")
    use_split = split and camera in ("auto", "split")

    cam_iter = None
    if not use_split:
        cam_path = dataset / "videos" / str(ep) / ("left.mp4" if camera == "left"
                                                   else "rgb.mp4")
        if not cam_path.exists():
            raise FileNotFoundError(f"ep {ep}: no camera video at {cam_path}")
        cam_iter = decode(cam_path)

    container = av.open(str(out), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(int(round(fps)), 1))
    stream.width, stream.height = 3 * PANEL_W, PANEL_H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "veryfast"}
    n_written = 0
    try:
        for i, frame in enumerate(decode(traj_path)):
            if use_split:
                cam, traj = frame[:, :PANEL_W], frame[:, PANEL_W:]
            else:
                traj = frame if frame.shape[1] == PANEL_W else frame[:, PANEL_W:]
                raw = next(cam_iter, None)
                cam = (np.full((PANEL_H, PANEL_W, 3), BG, np.uint8) if raw is None
                       else letterbox(raw))
            lines = src.header(ep, i, n_frames)
            t, valid, note = src.frame(i)
            if t is None:
                mid = missing_panel(note, domain=domain, cmap=cmap, z_near=src.z_near,
                                    z_far=src.z_far, lines=lines)
            else:
                mid = depth_panel(t, valid, domain=domain, cmap=cmap, z_near=src.z_near,
                                  z_far=src.z_far, lines=lines, note=note)
            comp = np.hstack([cam, mid, traj])
            comp[:, PANEL_W] = (70, 70, 70)
            comp[:, 2 * PANEL_W] = (70, 70, 70)
            for packet in stream.encode(av.VideoFrame.from_ndarray(comp, format="bgr24")):
                container.mux(packet)
            n_written += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    if src.n_source and n_written != src.n_source:
        print(f"  ep {ep}: NOTE {n_written} video frames but depth covers "
              f"{src.n_source}", flush=True)
    rule = colour_rule_json(domain, cmap, src.z_near, src.z_far)
    rule.update({"written_by": "data_collection/make_depth_trajectory_video.py",
                 "depth_source": src.kind, "depth_artifact": str(src.path),
                 "trajectory_video": str(traj_path), "frames": n_written,
                 "panels": ["camera", "depth", "trajectory"],
                 "depth_store_attrs": {k: v for k, v in src.attrs.items()
                                       if k != "dataset_plan"}})
    out.with_suffix(".colour_rule.json").write_text(json.dumps(rule, indent=2))
    return out


# -------------------------------------------------------------------------------- cli

def parse_episode_spec(spec: str, available):
    if spec.strip() in ("all", ""):
        return list(available)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [e for e in out if e in set(available)] or out


def episodes_on_disk(slam_root: Path):
    return sorted(int(p.name) for p in slam_root.iterdir()
                  if p.is_dir() and p.name.isdigit())


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="the UMI dataset root")
    p.add_argument("--slam-root", type=Path, default=None)
    p.add_argument("--episodes", default="all", help="all | 0,5,10-12")
    p.add_argument("--depth-source", choices=("fs", "obs"), default="fs",
                   help="fs: depth/<ep>/depth.zarr, FoundationStereo's own output "
                        "(default). obs: the 224x224 tensor training consumed")
    p.add_argument("--depth-store", type=Path, default=None,
                   help="--depth-source obs: path to dataset_depth.zarr[.zip]")
    p.add_argument("--domain", choices=("obs", "metric"), default="obs",
                   help="colour ramp: obs = inverse depth, exactly the training "
                        "normalisation (default); metric = linear in metres")
    p.add_argument("--cmap", choices=sorted(CMAPS), default="turbo")
    p.add_argument("--camera", choices=("auto", "split", "rgb", "left"), default="auto",
                   help="left panel: auto/split reuse trajectory.mp4's own camera panel; "
                        "rgb/left decode videos/<ep>/*.mp4 instead (left is the mono the "
                        "depth came from, but UNRECTIFIED, so it is not pixel-aligned)")
    p.add_argument("--out-name", default="trajectory_depth.mp4")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write <out-dir>/<ep>_<out-name> instead of beside trajectory.mp4")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset = args.dataset
    slam_root = args.slam_root or (dataset / "slam")
    if not slam_root.is_dir():
        print(f"no slam root: {slam_root}", file=sys.stderr)
        return 2
    episodes = parse_episode_spec(args.episodes, episodes_on_disk(slam_root))

    shared = None
    if args.depth_source == "obs":
        shared = ObsDepth(dataset, args.depth_store, args.domain)

    done = skipped = failed = 0
    for ep in episodes:
        ep_dir = slam_root / str(ep)
        if not (ep_dir / "trajectory.mp4").exists():
            print(f"  ep {ep}: no trajectory.mp4 (run UMI_Underwater/demonstration_"
                  f"processing/visualize_trajectory.py); skipping", flush=True)
            skipped += 1
            continue
        out = (args.out_dir / f"{ep}_{args.out_name}") if args.out_dir \
            else (ep_dir / args.out_name)
        if out.exists() and not args.overwrite:
            print(f"  ep {ep}: exists, skipping ({out})", flush=True)
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            src = (shared.for_episode(ep) if shared is not None
                   else FsDepth(dataset, ep, args.domain))
            render_episode(dataset, slam_root, ep, src, out, camera=args.camera,
                           domain=args.domain, cmap=args.cmap, crf=args.crf)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  ep {ep}: {exc}", flush=True)
            failed += 1
            continue
        done += 1
        print(f"  ep {ep}: {out}", flush=True)
    print(f"==> wrote {done} video(s), skipped {skipped}, failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
