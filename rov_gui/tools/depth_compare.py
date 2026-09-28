#!/usr/bin/env python3
"""
depth_compare.py — land depth and water depth on ONE colour scale.

Why this exists
---------------
On 2026-09-03 a land-trained diffusion policy was flown underwater and produced
an indecisive reference. The first debugging step agreed with the collaborator
(Slack, 2026-09-04) was: *match the depth maps as closely as possible and check
they are visually similar — and when you overlay them, make sure they share the
same colour mapping, just to be explicitly sure.*

That check could not be made, because until this file existed **no two depth
pictures in this project shared a colour rule.** Four renderers, four
conventions, all of the same physical quantity:

    land recorder preview     TURBO, LINEAR mm, warm = FAR,  fixed [MinZ, 2000 mm]
                              (umi_handheld/record.py:529-536)
    land FoundationStereo mp4 TURBO, LINEAR m,  warm = NEAR, fixed [0.2, 3.0 m]
                              (the repo-root umi_ep0_foundationstereo.mp4; its
                              generator was a scratchpad script, not in the tree)
    water device-stereo panel TURBO, LINEAR mm, warm = FAR,  fixed [300, 6000 mm]
                              (rov_gui/imaging.py:105-125)
    water FoundationStereo    JET,   warm = NEAR, and — the one that really bit —
      panel                   PER-FRAME 2/98-percentile auto range PLUS a PER-FRAME
                              64-knot quantile histogram equalisation
                              (rov_gui/backends/hardware.py:3297-3304,
                               rov_gui/imaging.py:173-191)
                              FIXED 2026-09-06: that panel now paints with the
                              rule below by default (`--fstereo-palette umi`),
                              out of rov_gui/depth_colour.py — the same function
                              this file calls. The adaptive picture is still
                              reachable as `--fstereo-palette adaptive`.

The last one is not merely a different palette: it is **scene-adaptive**, so the
same physical distance gets a different colour in every frame and in every run.
Laying that beside a fixed-ramp land video and concluding "the depth looks
different" measures the renderers, not the depth.

So this module is ONE renderer, used for both ends, with every knob nailed down:

    * fixed endpoints, taken from the training store's own ``.zattrs['obs']``
      (z_near 0.20 m, z_far 3.00 m) — never from the frame,
    * no auto-range, no histogram equalisation, ever,
    * one colormap for both sides (``--cmap``, default TURBO),
    * warm = NEAR in both domains, stated on the image,
    * invalid (no measurement) is BLACK and is distinguished from far,
    * the scale, its endpoints, the domain and the colormap name are BURNED
      INTO the picture, so a screenshot cannot be misread later,
    * and the same facts are written to ``colormap.json`` beside the output.

Two domains, and the difference matters
---------------------------------------
``--domain obs`` (default) renders the 224x224 uint8 tensor the policy actually
consumes: channel 0 is INVERSE depth normalised over [z_near, z_far], near =
large. This is the apples-to-apples domain, because the land builder
(``umi_handheld/build_dp_depth_zarr.py``) and the live builder
(``rov_gui/perception/policy_obs.py``) construct it by the same arithmetic —
parity-tested to bit-identity (``rov_gui/tests/test_policy_obs.py``). A pixel
value therefore means the same distance on both sides BY CONSTRUCTION.

``--domain metric`` renders millimetres on a fixed linear ramp instead. Use it
to answer "are the numbers also different", which is a different question from
"does the picture look different" and needs a linear axis to be legible.

Note that the land mp4 already circulated (umi_ep0_foundationstereo.mp4) is
LINEAR in metres while the obs is INVERSE, over the SAME endpoints. A 1.00 m
surface sits at palette position 0.29 in that video and 0.14 in the obs. They
are not the same picture even before the colormap differs.

Where the water side comes from
-------------------------------
Two producers write the ``policy_obs/`` layout this tool reads:

``--record-depth`` on the station saves what the POLICY consumed during a real
run. Before it existed a run recorded no depth at all — not the millimetre maps
and not the stereo pairs, so it could not even be regenerated offline
(``rov_gui/backends/hardware.py:3244-3352``, verified against
data/20260903/0903_183555/).

``rov_gui/tools/depth_capture_umi.py`` captures from the OAK-D-W that collected
the LAND demonstrations, now that it is mounted on the vehicle. That is the
stronger comparison of the two, because it removes the instrument from the
experiment entirely: same sensors, same EEPROM calibration (checked live,
max|dK| 0.0000 px on 2026-09-06), same rectifier object, same FoundationStereo
settings, same warp. What differs is the scene, the medium and the mount.

Usage
-----
    # land only — the training obs, one episode, as a strip and an mp4
    python -m rov_gui.tools.depth_compare land --episode 0 --out /tmp/land

    # water only — a run recorded with --record-depth
    python -m rov_gui.tools.depth_compare water \\
        --run data/20260910/0910_141530 --out /tmp/water

    # both, side by side, one colour rule, aligned by hand at the grasp
    python -m rov_gui.tools.depth_compare pair \\
        --episode 0 --land-frame 250 \\
        --run data/20260910/0910_141530_observe --water-frame impact --out /tmp/pair

    # the numbers, not the picture — including the near-field row, which is
    # where the gripper habitually sits and the one geometry check that a
    # matched colour scale cannot make for you
    python -m rov_gui.tools.depth_compare stats --episode 0 --run data/20260910/...

Environment
-----------
Runs anywhere numpy + cv2 exist. The land zarr is read WITHOUT the zarr package
by reusing ``dp_policy_offline.ZarrV2Store`` / ``TrainingStore``, which shells
out to a numcodecs-bearing interpreter when this one has no blosc — the same
mechanism, and the same reason, as that tool (``rovgui-pose`` and ``robust``
both lack zarr, and the repo root has a ``zarr/`` DATA directory that shadows
the package as an empty namespace module).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rov_gui.tools.dp_policy_offline import (  # noqa: E402
    BloscDecoder, TrainingStore, DEFAULT_DATA, DEFAULT_DATA_DIR)

# =============================================================================
# the colour rule — the whole point of this file
# =============================================================================

# The colour rule itself now lives in ``rov_gui/depth_colour.py`` and is
# RE-EXPORTED here, unchanged. It moved on 2026-09-06 so the live station could
# import it without dragging this file's zarr/decoder machinery into the GUI
# process — the FoundationStereo panel paints with the same function as these
# pictures now, which is what makes a screenshot comparable with a land video.
# There is still exactly one implementation; this is where callers and the docs
# already look for it.
from rov_gui.depth_colour import (  # noqa: E402,F401
    CMAPS, DEFAULT_CMAP, Z_FAR_M, Z_NEAR_M, _cv2, colorize, colour_rule_json,
    depth_mm_to_bgr, mm_to_t_valid, obs_to_metres, obs_to_t_valid,
    palette_pos, palette_pos_inverse, rule_text)


# =============================================================================
# rendering: the picture, and the legend that cannot lie about it
# =============================================================================

_BAR_W = 26
_PAD = 10
_LABEL_H = 34


def _put(img, text, org, scale=0.42, colour=(235, 235, 235), thick=1):
    cv2 = _cv2()
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def colour_bar(height: int, domain: str, cmap: str, z_near: float,
               z_far: float, ticks_m: Sequence[float] | None = None
               ) -> np.ndarray:
    """The legend, drawn by INVERTING ``palette_pos`` — never by an independent
    formula, so a tick cannot claim a distance the picture does not mean."""
    cv2 = _cv2()
    ramp = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    bar = colorize(np.repeat(ramp, _BAR_W, axis=1),
                   np.ones((height, _BAR_W), bool), cmap)
    panel = np.zeros((height, _BAR_W + 62, 3), np.uint8)
    panel[:, :_BAR_W] = bar
    if ticks_m is None:
        ticks_m = [z_near, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, z_far]
    for z in ticks_m:
        if not (z_near - 1e-9 <= z <= z_far + 1e-9):
            continue
        t = float(palette_pos(np.array([z]), domain, z_near, z_far)[0])
        y = int(round((1.0 - t) * (height - 1)))
        cv2.line(panel, (0, y), (_BAR_W - 1, y), (255, 255, 255), 1)
        _put(panel, f"{z:.2f}", (_BAR_W + 3, min(height - 2, y + 4)), 0.35)
    return panel


def render(t: np.ndarray, valid: np.ndarray, *, title: str, domain: str,
           cmap: str, z_near: float = Z_NEAR_M, z_far: float = Z_FAR_M,
           subtitle: str = "", scale: int = 2) -> np.ndarray:
    """One depth frame -> a labelled BGR panel with its legend attached.

    Everything a later reader needs to know about the colour rule is IN the
    image: domain, colormap, endpoints, and which end is warm. That is
    deliberate — the failure this whole file exists to prevent is two pictures
    being compared months later with nobody able to say what either scale was.
    """
    cv2 = _cv2()
    body = colorize(t, valid, cmap)
    if scale > 1:
        body = cv2.resize(body, (body.shape[1] * scale, body.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)
    h, w = body.shape[:2]
    bar = colour_bar(h, domain, cmap, z_near, z_far)
    canvas = np.zeros((h + _LABEL_H + _PAD, w + bar.shape[1] + _PAD * 2, 3),
                      np.uint8)
    canvas[:] = (24, 24, 24)
    canvas[_LABEL_H:_LABEL_H + h, _PAD:_PAD + w] = body
    canvas[_LABEL_H:_LABEL_H + h, _PAD * 2 + w:_PAD * 2 + w + bar.shape[1]] = bar
    _put(canvas, title, (_PAD, 16), 0.50, (255, 255, 255), 1)
    rule = (f"{domain} | {cmap.upper()} | warm=NEAR | "
            f"{z_near:.2f}-{z_far:.2f} m | black=no measurement")
    _put(canvas, rule if not subtitle else f"{rule}   {subtitle}",
         (_PAD, 30), 0.36, (170, 200, 255), 1)
    return canvas


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    out = np.zeros((h, left.shape[1] + right.shape[1] + 6, 3), np.uint8)
    out[:] = (24, 24, 24)
    out[:left.shape[0], :left.shape[1]] = left
    out[:right.shape[0], left.shape[1] + 6:] = right
    return out


# =============================================================================
# sources
# =============================================================================

class LandSource:
    """Frames of the LAND training obs, straight out of the store."""

    def __init__(self, path: str | Path, helper_python: Optional[str] = None):
        p = Path(path)
        if not p.exists():
            alt = Path(DEFAULT_DATA_DIR)
            if str(p) == DEFAULT_DATA and alt.exists():
                p = alt
            else:
                raise FileNotFoundError(f"land store not found: {p}")
        self.path = p
        self.decoder = BloscDecoder(helper_python)
        self.store = TrainingStore(p, self.decoder)
        obs_attrs = self.store.attrs.get("obs", {})
        self.z_near = float(obs_attrs.get("z_near_m", Z_NEAR_M))
        self.z_far = float(obs_attrs.get("z_far_m", Z_FAR_M))
        if abs(self.z_near - Z_NEAR_M) > 1e-9 or abs(self.z_far - Z_FAR_M) > 1e-9:
            print(f"NOTE: store endpoints {self.z_near}/{self.z_far} m differ "
                  f"from this tool's defaults {Z_NEAR_M}/{Z_FAR_M}; using the "
                  f"store's, and the water side must match or the comparison "
                  f"is meaningless.", file=sys.stderr)
        self.episodes = self.store.episodes()

    def describe(self) -> dict:
        a = self.store.attrs
        return {"path": str(self.path), "n_frames": self.store.n,
                "n_episodes": len(self.episodes),
                "obs": a.get("obs", {}), "depth_source": a.get("depth_source", {}),
                "decoder": self.decoder.backend}

    def frame_indices(self, episode: Optional[int], every: int,
                      limit: Optional[int]) -> list[int]:
        if episode is None:
            lo, hi = 0, self.store.n
        else:
            if not (0 <= episode < len(self.episodes)):
                raise IndexError(f"episode {episode} of {len(self.episodes)}")
            lo, hi = self.episodes[episode]
        idx = list(range(lo, hi, max(1, every)))
        return idx[:limit] if limit else idx

    def obs(self, idx: Sequence[int]) -> np.ndarray:
        return self.store.depth_frames(list(idx))


class WaterSource:
    """Frames of a RECORDED underwater run.

    Reads what ``--record-depth`` writes: ``policy_obs/index.csv`` plus one
    lossless PNG per tick (obs 224x224x3 uint8, and optionally the uint16 mm
    map it was built from). Nothing here decodes an mp4: the station's screen
    recording is the ALREADY-COLOURISED, downscaled picture and is not depth.
    """

    OBS_DIR = "policy_obs"

    def __init__(self, run: str | Path):
        self.run = Path(run)
        self.dir = self.run / self.OBS_DIR
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"no recorded depth in {self.run}.\n"
                f"  Expected {self.dir}/index.csv — written only when the run "
                f"was flown with `--record-depth`.\n"
                f"  A run without it has NO depth on disk at all: the uint16 "
                f"maps and the raw stereo pairs are never persisted, so the "
                f"depth cannot be regenerated offline either. The ui_*.mp4 is "
                f"a recording of the colourised screen, not of depth.")
        rows = (self.dir / "index.csv").read_text().strip().splitlines()
        if len(rows) < 2:
            # An index with only its header is what a recorder killed between
            # opening the folder and its first successful write leaves behind.
            # Saying so beats an IndexError two frames deeper.
            raise ValueError(
                f"{self.dir / 'index.csv'} has a header and no rows: the "
                f"recorder opened the folder but never wrote a frame. Check "
                f"the run meta's policy.depth_record.counters (and its `why`) "
                f"for what stopped it.")
        head = rows[0].split(",")
        self.rows = [dict(zip(head, r.split(","))) for r in rows[1:]]
        meta = self.dir / "meta.json"
        if not meta.exists():
            # Do NOT silently fall back to this module's endpoints. They happen
            # to match the shipped config today, so the metres would be right
            # by coincidence — and wrong, with nothing saying so, for any run
            # flown against a checkpoint with different ones.
            raise FileNotFoundError(
                f"{meta} is missing, so this recording does not state the "
                f"z_near/z_far it was normalised with, and its pixels cannot "
                f"be turned into distances. A recorder killed before close() "
                f"leaves this; the frames are still readable by hand.")
        self.meta = json.loads(meta.read_text())
        obs = self.meta.get("obs") or {}
        if obs.get("z_near_m") is None or obs.get("z_far_m") is None:
            raise ValueError(f"{meta} carries no obs endpoints (z_near_m / "
                             f"z_far_m); the recording cannot be scaled.")
        self.z_near = float(obs["z_near_m"])
        self.z_far = float(obs["z_far_m"])

    def describe(self) -> dict:
        return {"run": str(self.run), "n_frames": len(self.rows),
                "meta": self.meta}

    def obs(self, idx: Sequence[int]) -> np.ndarray:
        cv2 = _cv2()
        out = []
        for i in idx:
            p = self.dir / self.rows[int(i)]["obs_png"]
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None:
                raise FileNotFoundError(f"unreadable obs frame: {p}")
            out.append(im)
        return np.stack(out, axis=0)

    def depth_mm(self, i: int) -> Optional[np.ndarray]:
        cv2 = _cv2()
        name = self.rows[int(i)].get("depth_png", "")
        if not name:
            return None
        p = self.dir / name
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return None if im is None else im.astype(np.uint16)

    def frame_indices(self, every: int, limit: Optional[int]) -> list[int]:
        idx = list(range(0, len(self.rows), max(1, every)))
        return idx[:limit] if limit else idx


# =============================================================================
# outputs
# =============================================================================

def write_strip(panels: Sequence[np.ndarray], path: Path, cols: int = 6) -> None:
    cv2 = _cv2()
    if not panels:
        return
    h, w = panels[0].shape[:2]
    # Never pad past what there is: a one-panel sheet laid out in six columns
    # is five sixths dark canvas, which reads as a rendering failure.
    cols = max(1, min(int(cols), len(panels)))
    rows = (len(panels) + cols - 1) // cols
    sheet = np.full((rows * h, cols * w, 3), 24, np.uint8)
    for k, p in enumerate(panels):
        r, c = divmod(k, cols)
        sheet[r * h:r * h + p.shape[0], c * w:c * w + p.shape[1]] = p
    cv2.imwrite(str(path), sheet)


def write_mp4(panels: Sequence[np.ndarray], path: Path, fps: float) -> bool:
    cv2 = _cv2()
    if not panels:
        return False
    h, w = panels[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        return False
    for p in panels:
        vw.write(p if p.shape[:2] == (h, w) else
                 cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST))
    vw.release()
    return True


def _pct_table(name: str, z: np.ndarray) -> str:
    v = z[np.isfinite(z)]
    if v.size == 0:
        return f"{name:<10s} (no valid pixels)"
    q = lambda p: float(np.percentile(v, p))
    return (f"{name:<10s} valid {100.0 * v.size / z.size:5.1f}%   "
            f"p1 {q(1):5.2f}  p10 {q(10):5.2f}  p50 {q(50):5.2f}  "
            f"p90 {q(90):5.2f}  p99 {q(99):5.2f}   mean {v.mean():5.2f} m")


def near_row(z: np.ndarray, pct: float = 8.0) -> float:
    """Row centroid of the NEAREST ``pct``% of valid pixels, 0 = top, 1 = bottom.

    Depth-only policies have no colour to tell the gripper from the scene; what
    they have is a large near-field blob in a habitual place. In the land
    demonstrations that blob is the handheld fingers entering from the BOTTOM
    edge — every training frame, without exception. If the deployment rig puts
    it somewhere else, every observation is out of distribution in a way no
    amount of range matching can repair, and this one number says so in a form
    you can re-check in a minute after moving a bracket.
    """
    v = np.isfinite(z)
    if not v.any():
        return float("nan")
    m = v & (z <= np.nanpercentile(z[v], pct))
    if not m.any():
        return float("nan")
    return float(np.nonzero(m)[0].mean() / max(1, z.shape[0] - 1))


def _near_row_line(name: str, zs: np.ndarray) -> str:
    r = np.array([near_row(z) for z in zs], float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return f"{'':10s} near-field row: (none)"
    where = "UPPER half" if r.mean() < 0.5 else "LOWER half"
    return (f"{'':10s} near-field row {r.mean():.3f} +- {r.std():.3f} "
            f"(min {r.min():.3f} max {r.max():.3f}) -> {where}"
            f"   [0 = top edge, 1 = bottom edge]")


# =============================================================================
# commands
# =============================================================================

def _panels_from_obs(obs: np.ndarray, idx: Sequence[int], *, label: str,
                     domain: str, cmap: str, z_near: float, z_far: float,
                     scale: int) -> list[np.ndarray]:
    out = []
    for k, i in enumerate(idx):
        o = obs[k]
        if domain == "obs":
            t, valid = obs_to_t_valid(o)
        else:
            z = obs_to_metres(o, z_near, z_far)
            valid = np.isfinite(z)
            t = palette_pos(np.where(valid, z, z_far), domain, z_near, z_far)
        out.append(render(t, valid, title=f"{label}  frame {i}", domain=domain,
                          cmap=cmap, z_near=z_near, z_far=z_far, scale=scale))
    return out


def cmd_land(a) -> int:
    src = LandSource(a.data, a.helper_python)
    print(json.dumps(src.describe(), indent=2)[:1500])
    idx = src.frame_indices(a.episode, a.every, a.limit)
    print(f"land: {len(idx)} frames from "
          f"{'episode ' + str(a.episode) if a.episode is not None else 'the whole store'}")
    obs = src.obs(idx)
    panels = _panels_from_obs(obs, idx, label="LAND (training obs)",
                              domain=a.domain, cmap=a.cmap, z_near=src.z_near,
                              z_far=src.z_far, scale=a.scale)
    return _emit(panels, a, src.z_near, src.z_far, {"land": src.describe()})


def cmd_water(a) -> int:
    src = WaterSource(a.run)
    print(json.dumps(src.describe(), indent=2)[:1500])
    idx = src.frame_indices(a.every, a.limit)
    print(f"water: {len(idx)} frames from {src.run}")
    obs = src.obs(idx)
    panels = _panels_from_obs(obs, idx, label="WATER (live obs)",
                              domain=a.domain, cmap=a.cmap, z_near=src.z_near,
                              z_far=src.z_far, scale=a.scale)
    return _emit(panels, a, src.z_near, src.z_far, {"water": src.describe()})


def cmd_pair(a) -> int:
    land = LandSource(a.data, a.helper_python)
    water = WaterSource(a.run)
    if abs(land.z_near - water.z_near) > 1e-9 or abs(land.z_far - water.z_far) > 1e-9:
        print(f"REFUSING: land endpoints [{land.z_near}, {land.z_far}] m and "
              f"water endpoints [{water.z_near}, {water.z_far}] m differ. The "
              f"two pictures would not be comparable and this tool will not "
              f"draw them as if they were.", file=sys.stderr)
        return 2
    li = land.frame_indices(a.episode, a.every, a.limit)
    wi = water.frame_indices(a.every, a.limit)
    if a.land_frame is not None:
        li = [int(a.land_frame)]
    if a.water_frame is not None:
        wi = [int(a.water_frame)]
    n = min(len(li), len(wi))
    if n == 0:
        print("nothing to pair", file=sys.stderr)
        return 2
    li, wi = li[:n], wi[:n]
    lp = _panels_from_obs(land.obs(li), li, label="LAND (training obs)",
                          domain=a.domain, cmap=a.cmap, z_near=land.z_near,
                          z_far=land.z_far, scale=a.scale)
    wp = _panels_from_obs(water.obs(wi), wi, label="WATER (live obs)",
                          domain=a.domain, cmap=a.cmap, z_near=water.z_near,
                          z_far=water.z_far, scale=a.scale)
    panels = [side_by_side(l, w) for l, w in zip(lp, wp)]
    return _emit(panels, a, land.z_near, land.z_far,
                 {"land": land.describe(), "water": water.describe(),
                  "pairing": "BY INDEX ONLY — these two runs are not time-aligned. "
                             "Use --land-frame/--water-frame to align them by "
                             "hand at the grasp, which is the only landmark the "
                             "two share."})


def cmd_stats(a) -> int:
    """The numbers, so 'do they look different' can be answered arithmetically."""
    out = {}
    print(f"\ndepth distributions, in METRES, over the obs grid")
    print(f"endpoints {Z_NEAR_M}-{Z_FAR_M} m; 'valid' excludes no-measurement pixels\n")
    if a.data:
        try:
            land = LandSource(a.data, a.helper_python)
            idx = land.frame_indices(a.episode, a.every, a.limit or 200)
            z = np.stack([obs_to_metres(o, land.z_near, land.z_far)
                          for o in land.obs(idx)])
            print(_pct_table("LAND", z))
            print(_near_row_line("LAND", z))
            u = np.unique(land.obs(idx)[..., 0])
            print(f"{'':10s} ch0 u8 range {int(u.min())}..{int(u.max())} of 0..255 "
                  f"({100.0 * (u.max() - u.min()) / 255:.0f}% of the byte used); "
                  f"one u8 step = {abs(palette_pos_inverse(0.5, 'obs') - palette_pos_inverse(0.5 + 1/255, 'obs')) * 1000:.0f} mm "
                  f"at the palette midpoint")
            out["land"] = {"n_frames": len(idx), "path": str(land.path)}
        except Exception as e:                      # noqa: BLE001
            print(f"LAND       unavailable: {e}")
    if a.run:
        try:
            water = WaterSource(a.run)
            idx = water.frame_indices(a.every, a.limit or 200)
            z = np.stack([obs_to_metres(o, water.z_near, water.z_far)
                          for o in water.obs(idx)])
            print(_pct_table("WATER", z))
            print(_near_row_line("WATER", z))
            out["water"] = {"n_frames": len(idx), "run": str(water.run)}
        except Exception as e:                      # noqa: BLE001
            print(f"WATER      unavailable: {e}")
    print()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


def _emit(panels, a, z_near: float, z_far: float, sources: dict) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rule = colour_rule_json(a.domain, a.cmap, z_near, z_far)
    (out / "colormap.json").write_text(json.dumps(
        {"colour_rule": rule, "sources": sources,
         "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    if a.strip:
        write_strip(panels, out / "strip.png", a.cols)
        print(f"wrote {out / 'strip.png'}  ({len(panels)} panels)")
    if a.video:
        ok = write_mp4(panels, out / "depth.mp4", a.fps)
        print(f"wrote {out / 'depth.mp4'}" if ok else
              "VideoWriter refused to open — strip only")
    if a.frames:
        cv2 = _cv2()
        fd = out / "frames"
        fd.mkdir(exist_ok=True)
        for k, p in enumerate(panels):
            cv2.imwrite(str(fd / f"{k:05d}.png"), p)
        print(f"wrote {len(panels)} frames to {fd}")
    print(f"wrote {out / 'colormap.json'} — the colour rule, so this picture "
          f"stays readable after the session that made it")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="depth_compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, *, needs_land=False, needs_water=False):
        q.add_argument("--domain", choices=("obs", "metric"), default="obs",
                       help="obs = the normalised inverse-depth tensor the "
                            "policy consumes (apples-to-apples, default); "
                            "metric = a linear millimetre ramp, for comparing "
                            "the NUMBERS rather than the picture")
        q.add_argument("--cmap", choices=sorted(CMAPS), default="turbo")
        q.add_argument("--every", type=int, default=10, metavar="N",
                       help="take every Nth frame (default: %(default)s)")
        q.add_argument("--limit", type=int, default=None, metavar="N")
        q.add_argument("--scale", type=int, default=2,
                       help="nearest-neighbour magnification of the 224 grid")
        q.add_argument("--out", default="/tmp/depth_compare", metavar="DIR")
        q.add_argument("--strip", action="store_true", default=True)
        q.add_argument("--no-strip", dest="strip", action="store_false")
        q.add_argument("--cols", type=int, default=6)
        q.add_argument("--video", action="store_true",
                       help="also write depth.mp4")
        q.add_argument("--frames", action="store_true",
                       help="also write every panel as its own png")
        q.add_argument("--fps", type=float, default=10.0)
        if needs_land:
            q.add_argument("--data", default=DEFAULT_DATA, metavar="ZARR")
            q.add_argument("--episode", type=int, default=None)
            q.add_argument("--helper-python", default=None,
                           help="interpreter with numcodecs, for blosc chunks")
        if needs_water:
            q.add_argument("--run", required=True, metavar="RUNDIR",
                           help="a run flown with --record-depth")

    q = sub.add_parser("land", help="render the LAND training obs")
    common(q, needs_land=True)
    q.set_defaults(func=cmd_land)

    q = sub.add_parser("water", help="render a RECORDED underwater run's obs")
    common(q, needs_water=True)
    q.set_defaults(func=cmd_water)

    q = sub.add_parser("pair", help="both, side by side, one colour rule")
    common(q, needs_land=True, needs_water=True)
    q.add_argument("--land-frame", type=int, default=None,
                   help="a single land frame index (align by hand at the grasp)")
    q.add_argument("--water-frame", type=int, default=None)
    q.set_defaults(func=cmd_pair)

    q = sub.add_parser("stats", help="the distributions, in metres")
    q.add_argument("--data", default=DEFAULT_DATA, metavar="ZARR")
    q.add_argument("--episode", type=int, default=None)
    q.add_argument("--run", default=None, metavar="RUNDIR")
    q.add_argument("--every", type=int, default=10)
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--helper-python", default=None)
    q.add_argument("--out", default=None, metavar="JSON")
    q.set_defaults(func=cmd_stats)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
