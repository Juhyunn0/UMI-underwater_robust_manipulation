#!/usr/bin/env python3
"""dp_policy_offline.py — the policy stack against its own training data.

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tools/dp_policy_offline.py gate \
        [--action-repr pos_yaw_width|pos_rpy_width|pose10d]
    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tools/dp_policy_offline.py replay \
        [--val-only [--val-stride 4]] [--ckpt ...]
    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tools/dp_policy_offline.py noise

Three questions that must be answered BEFORE the first pool run, each with a
number rather than an opinion (spec v2 §B tests, agent C):

  gate    Would the safety filter even let the DEMONSTRATIONS through? The
          ground-truth action chunks of the training set (16 knots @ 66.7 ms,
          built exactly like ``diffusion_policy/common/sampler.py`` +
          ``umi_dataset.py``) are composed into ``PlanMsg`` with
          ``policy_frames.compose_plan`` (identity anchor) and pushed through
          ``plan_stream.PlanFilter`` with the DEFAULT ``policy:`` limits, on
          the raw 1/15 s grid and on the 0.2 s grid of spec A5. Prints the
          accept / clip / reject counts, the ``need`` (time-dilation factor
          the kinematic gate would want) distribution, the knot jitter and
          the peak speed / accel — the evidence behind A5's grid choice and
          A21's caps, or against them.
  replay  Run the REAL policy on training observations (dataset depth frames
          on the identity grid + ground-truth proprio rows) and compare its
          chunks with the ground truth per knot index (pos / yaw|rot / width
          RMS), plus the filter's verdict on the policy's own chunks and the
          inference time. This is the closed-form "does it reproduce the
          demos it was trained on" check; it says nothing about water.
          ``--val-only`` swaps the stride-8 whole-store selection (mostly
          TRAIN windows) for the HELD-OUT episodes of the training split:
          ``get_val_mask(75, 0.05, 42)`` = episodes 6/32/48/56, reimplemented
          VERBATIM from ``diffusion_policy/common/sampler.py``, every valid
          window start of those episodes at ``--val-stride``. It prints, per
          knot and overall, the de-normalised per-element pos MSE [m^2] and
          RMSE [mm] (the upstream ``val_action_mse_error_pos`` definition),
          yaw RMS [deg] (pos_yaw_width; rot RMS [deg] for pose10d, with the
          yaw DERIVED from the rot6d through the same encoder) beside the
          zero-predictor baseline ``sqrt(mean(gt_dyaw^2))`` and corr(pred,
          GT dyaw) — the sign check — and width RMSE [mm]. This is the
          acceptance number of the 2026-09-07 retrain (plan §검증 2). The
          KNOWN_ISSUES.md:33-46 ledger (old ckpt pos MSE 0.000956 m^2 =
          30.9 mm RMSE over "219 windows") had no committed script; 219 =
          31+76+63+49 is exactly the count of valid starts at stride 4 in
          those four episodes [유도: lengths 153/331/280/225 frames, 30-frame
          tail], which is why the default is ``--val-stride 4``;
          ``--val-stride 1`` is every valid start (869 windows). Re-measure
          BOTH checkpoints with this tool rather than quoting the ledger.
  noise   Proprio sensitivity: N(0, sigma) added to the relative-pose obs
          row (sigma 0 / 2 / 5 / 10 mm, the tag-fix noise band) with the
          diffusion noise held fixed — how much the emitted chunk moves per
          millimetre of proprio error.

Reading the zarr WITHOUT the zarr package
----------------------------------------
``rovgui-pose`` has no real zarr (the repo's ``zarr/`` data directory shadows
it as a namespace package) and no blosc / numcodecs either, and the "plain"
directory copy ``dataset_depth.zarr`` on disk turned out to be blosc-zstd
compressed exactly like the zip (checked 2026-09-02: identical ``.zarray``
compressor blocks). A zarr v2 store is just ``.zarray`` JSON + one file per
chunk, so :class:`ZarrV2Array` reads both layouts with ``zipfile`` /
``pathlib`` and hands the chunk bytes to whichever blosc decoder exists:
the ``blosc`` package, else ``numcodecs``, else a HELPER INTERPRETER that
has numcodecs (``$DP_OFFLINE_BLOSC_PYTHON``, or the training env ``umi2``
found under ``~/miniforge3/envs``) driven over a pipe in one batch per
request. The backend in use is printed; if none exists the tool says so and
stops — it never guesses at the container format.

Frames in the gate mode
-----------------------
The handheld's TCP frame is the camera optical frame (x right, y down,
z forward); body FRD is (x forward, y right, z down). ``compose_plan`` wants
``T_bt`` (body <- TCP); the gate uses the pure axis permutation with a ZERO
lever arm (``T_BT_GATE``): speed / accel norms are rotation-invariant, and
the real jaw offset (spec A3) only adds the lever-arm term, which is a
hardware question, not a dataset one. WHICH yaw the rate gate reads depends
on the label representation (``--action-repr``, 2026-09-07):

* ``pos_yaw_width`` (16, 5), the flown contract: the label's ``dyaw`` is the
  C3-mount-referenced yaw of the relative TCP rotation
  (``policy_frames.encode_pos_yaw`` = the training encoder
  ``umi/common/yaw_action.py``), and ``compose_plan`` decodes it against the
  ZERO anchor as ``yaw_k = 0 + dyaw_k`` — the gate sees the label itself,
  i.e. what a level ROV with the C3 at 43.3 deg would have to yaw.
* ``pose10d`` (16, 10), legacy: ``compose_plan`` takes the ZYX yaw of
  ``R_ned_tcp_k @ R_bt^T`` with ``R_bt`` = the PURE PERMUTATION of
  ``T_BT_GATE`` — the handheld's rotation about its own down axis, a
  different definition (no mount tilt).
* ``pos_rpy_width`` (16, 7), the 6-DoF variant (2026-09-26): columns 0:4
  and 6 are BIT-IDENTICAL to the 5-dim label (``policy_frames.encode_pos_rpy``
  shares the yaw arctan2 with ``encode_pos_yaw``), so its pos / yaw gate
  numbers equal the 5-dim ones exactly (a regression diff of the 5-dim
  rows is the check). The two extra columns are the ZYX roll / pitch of the
  same ``R_bt @ R_rel @ R_bt^T``; the gate composes them with
  ``track_rp=True`` (T1 clip at ``rp_max_deg``) and reports what the
  attitude gates (rp_mag reject, rp_rate dilation ``need`` term) would do
  to the DEMONSTRATIONS — the evidence behind ``rp_max_deg`` /
  ``pq_max_rad_s`` (both [예측] until this prints) — plus the GT |droll| /
  |dpitch| percentiles. The attitude limits used are printed beside the
  results like the others.

Position, speed and accel are identical in all three (``dp`` is bit-identical
to the legacy columns 0:3), so accept / clip / reject and ``need`` differ ONLY
through the yaw-rate gate (pose10d vs the two flyable reprs) and the attitude
gates (pos_rpy_width only): pose10d gate numbers are NOT comparable with the
other two and a change between them is not a regression. ``gate --action-repr
pose10d`` reproduces the pre-2026-09-07 output; ``--action-repr
pos_yaw_width`` (the default) prints the same pos / yaw numbers as before
2026-09-26.

Everything printed is a [측정] against the named zarr; the limits are the
spec's [예측]/[유도] defaults and are printed with the results so the two are
never confused.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.state import (ACTION_DIM_BY_REPR, ACTION_REPR_BY_DIM,  # noqa: E402
                           ACTION_REPR_POS_RPY_WIDTH, ACTION_REPR_POS_YAW_WIDTH,
                           ACTION_REPR_POSE10D, POLICY_ACTION_REPR)

DEFAULT_DATA = "/home/bdml/Desktop/data collection/dataset_depth.zarr.zip"
DEFAULT_DATA_DIR = "/home/bdml/Desktop/data collection/dataset_depth.zarr"

#: The v2 spec's DEFAULT ``policy:`` block, the part the filter needs
#: (DP_LIVE_POLICY_SPEC_V2 §B "MpcConfig.policy defaults"; every number
#: there is [예측] or [유도] as tagged in A21). If ``geometry.MpcConfig``
#: carries a ``policy`` block, drift from these is printed.
POLICY_DEFAULTS_V2 = {
    "v_max_m_s": 0.08, "a_max_m_s2": 0.20, "r_max_rad_s": 0.50,
    "anchor_max_m": 0.15, "jump_max_m": 0.06, "yaw_jump_max_deg": 8.0,
    "obs_max_age_s": 0.6, "knot_dt_s": 0.2,
    "workspace_box_ned": [[-1.0, -1.0, -0.5], [1.0, 1.0, 0.5]],
    "gripper_width_open_m": 0.069, "gripper_width_closed_m": 0.042,
    "clip_ratio_max": 3.0,
    # 6-DoF variant attitude limits (design D3, 2026-09-26; all [예측] --
    # this tool's ``gate --action-repr pos_rpy_width`` output is what turns
    # them into artefact-based numbers). Only read for 7-dim chunks.
    "rp_max_deg": 20.0, "rp_reject_deg": 30.0, "pq_max_rad_s": 0.35,
    "rp_jump_max_deg": 5.0,
}

#: body FRD <- TCP (camera optical) axis permutation, zero lever arm (see
#: module docstring): TCP x (right) -> body y, TCP y (down) -> body z,
#: TCP z (forward) -> body x.
T_BT_GATE = np.array([[0.0, 0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

KNOT_DTS = (1.0 / 15.0, 0.2)


# =============================================================================
# zarr v2 reader (no zarr package)
# =============================================================================
_HELPER_SRC = r"""
import struct, sys
import numcodecs
codec = numcodecs.Blosc()
r, w = sys.stdin.buffer, sys.stdout.buffer
n = struct.unpack("<I", r.read(4))[0]
for _ in range(n):
    L = struct.unpack("<Q", r.read(8))[0]
    out = bytes(memoryview(codec.decode(r.read(L))))
    w.write(struct.pack("<Q", len(out)))
    w.write(out)
w.flush()
"""


class BloscDecoder:
    """Decode blosc chunks with whatever this interpreter, or a helper one,
    can offer (module docstring). ``backend`` says which."""

    def __init__(self, helper_python: Optional[str] = None):
        self.backend = ""
        self._fn = None
        self._helper = None
        try:
            import blosc                                    # noqa: F401
            self._fn = lambda blobs: [blosc.decompress(b) for b in blobs]
            self.backend = f"blosc {blosc.__version__} (in-process)"
            return
        except ImportError:
            pass
        try:
            import numcodecs
            codec = numcodecs.Blosc()
            self._fn = lambda blobs: [bytes(memoryview(codec.decode(b)))
                                      for b in blobs]
            self.backend = f"numcodecs {numcodecs.__version__} (in-process)"
            return
        except ImportError:
            pass
        py = helper_python or os.environ.get("DP_OFFLINE_BLOSC_PYTHON") \
            or self._find_helper()
        if py is None:
            raise RuntimeError(
                "no blosc decoder: neither the 'blosc' nor the 'numcodecs' "
                "package is importable here, and no helper interpreter with "
                "numcodecs was found under ~/miniforge3/envs (set "
                "DP_OFFLINE_BLOSC_PYTHON=/path/to/python). The on-disk "
                "directory copy of the dataset is blosc-compressed too, so "
                "there is no uncompressed fallback to read.")
        self._helper = py
        self.backend = f"subprocess helper {py} (numcodecs)"

    @staticmethod
    def _find_helper() -> Optional[str]:
        envs = Path.home() / "miniforge3" / "envs"
        cands = [envs / n / "bin" / "python" for n in ("umi2", "umi", "umi_day",
                                                       "umi_day2")]
        if envs.is_dir():
            cands += sorted(p / "bin" / "python" for p in envs.iterdir()
                            if p.is_dir())
        seen = set()
        for c in cands:
            if not c.is_file() or c in seen:
                continue
            seen.add(c)
            try:
                r = subprocess.run([str(c), "-c", "import numcodecs"],
                                   capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                return str(c)
        return None

    def decode(self, blobs: Sequence[bytes]) -> List[bytes]:
        if not blobs:
            return []
        if self._fn is not None:
            return self._fn(blobs)
        payload = bytearray(struct.pack("<I", len(blobs)))
        for b in blobs:
            payload += struct.pack("<Q", len(b))
            payload += b
        r = subprocess.run([self._helper, "-c", _HELPER_SRC], input=bytes(payload),
                           capture_output=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"helper decoder failed: "
                               f"{r.stderr.decode(errors='replace')[-800:]}")
        out, buf, pos = [], r.stdout, 0
        for _ in blobs:
            (L,) = struct.unpack_from("<Q", buf, pos)
            pos += 8
            out.append(bytes(buf[pos:pos + L]))
            pos += L
        return out


class ZarrV2Store:
    """A zarr v2 ZipStore or DirectoryStore as a key -> bytes mapping."""

    def __init__(self, path):
        self.path = Path(path)
        self._zip = None
        if self.path.is_file():
            self._zip = zipfile.ZipFile(self.path)
            self._names = set(self._zip.namelist())
            self.kind = "zip"
        elif self.path.is_dir():
            self._names = None
            self.kind = "directory"
        else:
            raise FileNotFoundError(f"zarr store not found: {self.path}")

    def has(self, key: str) -> bool:
        if self._zip is not None:
            return key in self._names
        return (self.path / key).is_file()

    def read(self, key: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(key)
        return (self.path / key).read_bytes()

    def json(self, key: str) -> dict:
        return json.loads(self.read(key).decode("utf-8"))


class ZarrV2Array:
    """One array of a v2 store: chunked along axis 0 only (every array in
    the training store is — depth frames are (1, 224, 224, 3) chunks, the
    lowdim tables (8192, D)). ``prefetch`` decodes many chunks in ONE decoder
    call (the subprocess backend pays a process launch per call)."""

    def __init__(self, store: ZarrV2Store, name: str, decoder: BloscDecoder):
        self.store, self.name, self.decoder = store, name, decoder
        meta = store.json(f"{name}/.zarray")
        if meta.get("zarr_format") != 2:
            raise ValueError(f"{name}: not a zarr v2 array")
        self.shape = tuple(int(s) for s in meta["shape"])
        self.chunks = tuple(int(c) for c in meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.order = meta.get("order", "C")
        self.sep = meta.get("dimension_separator", ".")
        self.fill = meta.get("fill_value", 0)
        comp = meta.get("compressor")
        if comp is not None and comp.get("id") != "blosc":
            raise ValueError(f"{name}: compressor {comp.get('id')!r} unsupported "
                             f"(this reader knows blosc and None)")
        if meta.get("filters"):
            raise ValueError(f"{name}: filters unsupported")
        self.compressed = comp is not None
        if self.chunks[1:] != self.shape[1:]:
            raise ValueError(f"{name}: chunked beyond axis 0 "
                             f"({self.chunks} vs {self.shape}) — unsupported")
        self._cache: Dict[int, np.ndarray] = {}

    def _key(self, ci: int) -> str:
        return f"{self.name}/" + self.sep.join([str(ci)] + ["0"] * (len(self.shape) - 1))

    @property
    def n_chunks(self) -> int:
        return (self.shape[0] + self.chunks[0] - 1) // self.chunks[0]

    def prefetch(self, chunk_ids: Sequence[int]) -> None:
        want = sorted({int(c) for c in chunk_ids} - set(self._cache))
        if not want:
            return
        blobs, present = [], []
        for ci in want:
            k = self._key(ci)
            if self.store.has(k):
                blobs.append(self.store.read(k))
                present.append(ci)
            else:
                self._cache[ci] = np.full(self.chunks, self.fill, self.dtype)
        raw = self.decoder.decode(blobs) if self.compressed else blobs
        for ci, b in zip(present, raw):
            arr = np.frombuffer(b, dtype=self.dtype)
            self._cache[ci] = arr.reshape(self.chunks, order=self.order)

    def rows(self, i0: int, i1: int) -> np.ndarray:
        """``array[i0:i1]`` along axis 0 (chunk-padded rows trimmed)."""
        i0, i1 = max(0, int(i0)), min(self.shape[0], int(i1))
        c0 = self.chunks[0]
        ids = list(range(i0 // c0, (i1 - 1) // c0 + 1)) if i1 > i0 else []
        self.prefetch(ids)
        parts = []
        for ci in ids:
            lo, hi = ci * c0, min((ci + 1) * c0, self.shape[0])
            blk = self._cache[ci][: hi - lo]
            parts.append(blk[max(i0, lo) - lo: min(i1, hi) - lo])
        if not parts:
            return np.empty((0,) + self.shape[1:], self.dtype)
        return np.concatenate(parts, axis=0)

    def all(self) -> np.ndarray:
        return self.rows(0, self.shape[0])


class TrainingStore:
    """The training zarr's tables in memory + depth frames on demand."""

    LOWDIM = ("robot0_eef_pos", "robot0_eef_rot_axis_angle",
              "robot0_gripper_width", "robot0_demo_start_pose")

    def __init__(self, path, decoder: BloscDecoder):
        self.store = ZarrV2Store(path)
        self.attrs = self.store.json(".zattrs")
        self.decoder = decoder
        self.arrays = {k: ZarrV2Array(self.store, f"data/{k}", decoder)
                       for k in self.LOWDIM}
        self.depth = ZarrV2Array(self.store, "data/camera0_depth", decoder)
        self.episode_ends = ZarrV2Array(self.store, "meta/episode_ends",
                                        decoder).all().astype(int)
        self.tab = {k: a.all().astype(np.float64) for k, a in self.arrays.items()}
        self.n = int(self.arrays["robot0_eef_pos"].shape[0])
        self.fps = float(self.attrs.get("fps", 30.0))

    def episodes(self) -> List[Tuple[int, int]]:
        out, s = [], 0
        for e in self.episode_ends:
            out.append((s, int(e)))
            s = int(e)
        return out

    def depth_frames(self, idx: Sequence[int]) -> np.ndarray:
        """(n, 224, 224, 3) uint8 for frame indices (chunk == frame)."""
        idx = [int(i) for i in idx]
        self.depth.prefetch(idx)
        return np.stack([self.depth.rows(i, i + 1)[0] for i in idx], axis=0)


# =============================================================================
# ground truth exactly like sampler.py / umi_dataset.py
# =============================================================================
OBS_HORIZON, ACTION_HORIZON, DOWN_SAMPLE = 2, 16, 2


def rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    """Rodrigues, (…, 3) -> (…, 3, 3); scipy's from_rotvec().as_matrix()."""
    rv = np.asarray(rv, dtype=float)
    th = np.linalg.norm(rv, axis=-1)
    out = np.zeros(rv.shape[:-1] + (3, 3))
    for i in np.ndindex(rv.shape[:-1]):
        t = float(th[i])
        if t < 1e-12:
            out[i] = np.eye(3)
            continue
        k = rv[i] / t
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        out[i] = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)
    return out


def pose6_to_mat(pos: np.ndarray, rv: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    T = np.zeros(pos.shape[:-1] + (4, 4))
    T[..., :3, :3] = rotvec_to_mat(rv)
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def window_indices(store: TrainingStore, stride: int, min_windows: int
                   ) -> List[Tuple[int, int, int]]:
    """(current_idx, start, end) at ``stride`` inside each episode, only
    where a full 16-knot action fits (``action_padding: false`` in the cfg:
    sampler.py skips ``end < current + 15*2 + 1``). Halves the stride until
    ``min_windows`` are available."""
    need = (ACTION_HORIZON - 1) * DOWN_SAMPLE + 1
    while True:
        wins = []
        for s, e in store.episodes():
            for idx in range(s, e, stride):
                if e < idx + need:
                    break
                wins.append((idx, s, e))
        if len(wins) >= min_windows or stride <= 1:
            return wins
        stride = max(1, stride // 2)


#: The training split's held-out selection [스펙: diffusion_policy/config/task/
#: umi_depth.yaml + umi_depth_5d.yaml ``dataset.val_ratio`` / ``dataset.seed``].
VAL_RATIO, VAL_SEED = 0.05, 42
#: What ``get_val_mask(75, 0.05, 42)`` picks on the 75-episode store [측정
#: 2026-09-07: numpy 1.26 (robust) and 2.4 (rovgui-pose) agree]; asserted so a
#: Generator drift can never silently evaluate on TRAIN episodes.
VAL_EPISODES_75 = (6, 32, 48, 56)


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    """VERBATIM of ``diffusion_policy/common/sampler.py:get_val_mask`` — the
    same ``default_rng(seed).choice`` draw, so the episodes flagged here ARE
    the ones the trainer never saw."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def val_windows(store: TrainingStore, stride: int = 1
                ) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """((current_idx, start, end) of every valid window start at ``stride``
    inside the HELD-OUT episodes, those episode indices). Validity is
    :func:`window_indices`'s rule (a full 16-knot action must fit; the
    sampler's gripper threshold only flags, it never drops). ``stride`` 1 =
    all 869 windows of the 75-episode store; 4 = the 219-window set of
    KNOWN_ISSUES.md:33-46 (module docstring)."""
    n_ep = len(store.episode_ends)
    eps = [int(k) for k in np.flatnonzero(get_val_mask(n_ep, VAL_RATIO, VAL_SEED))]
    if n_ep == 75:
        assert tuple(eps) == VAL_EPISODES_75, (
            f"get_val_mask(75, {VAL_RATIO}, {VAL_SEED}) gave episodes {eps}, not "
            f"{list(VAL_EPISODES_75)}: this numpy's Generator.choice differs from "
            f"the training env's, so these would NOT be the held-out episodes")
    need = (ACTION_HORIZON - 1) * DOWN_SAMPLE + 1
    episodes = store.episodes()
    wins = []
    for k in eps:
        s, e = episodes[k]
        for idx in range(s, e, max(1, int(stride))):
            if e < idx + need:
                break
            wins.append((idx, s, e))
    return wins, eps


def gt_window(store: TrainingStore, idx: int, s: int, e: int, pf,
              start_noise: float = 0.0, rng=None,
              action_repr: str = ACTION_REPR_POS_YAW_WIDTH
              ) -> Tuple[dict, np.ndarray, dict]:
    """(lowdim obs dict, GT action (16, D), extras) for ``current_idx``.

    obs rows at ``[idx - 2, idx]`` clipped to ``[s, e-1]`` (sampler.py's
    ``idx_with_latency`` at integer positions = exact rows), relative to the
    LAST row (``convert_pose_mat_rep(..., 'relative')`` = inv(base) @ T);
    ``wrt_start`` = rotation of inv(T_start) @ T_i with the start pose row
    (training added N(0, 0.05) noise to it — ``start_noise`` reproduces
    that when asked; default 0 for a deterministic evaluation); action rows
    ``idx::2`` x 16, relative to the same base, width appended. The obs
    block does not depend on ``action_repr``.

    ``T_rel_k = inv(T_now) @ T_act_k`` with ``T_now`` the newest obs row (the
    same ``base_inv`` the obs uses). ``pos_yaw_width`` (D = 5, the flown
    contract): ``(dp, dyaw) = policy_frames.encode_pos_yaw(T_rel)`` — the
    training encoder ``umi/common/yaw_action.py`` verbatim, so ``dp`` is
    bit-identical to the legacy columns 0:3 and ``dyaw`` is the C3-mount-
    referenced yaw of the relative rotation — then ``[dp, dyaw, width]``.
    ``pose10d`` (D = 10, legacy): ``mat_to_pose10d(T_rel)`` + width, the
    pre-2026-09-07 path unchanged. ``pos_rpy_width`` (D = 7, 2026-09-26):
    ``(dp, dyaw, droll, dpitch) = policy_frames.encode_pos_rpy(T_rel)`` —
    the training encoder ``yaw_action.encode_pos_rpy`` verbatim — then
    ``[dp, dyaw, droll, dpitch, width]``; columns 0:4 and 6 are bit-identical
    to the 5-dim label.
    """
    pos, rv, width, start = (store.tab["robot0_eef_pos"],
                             store.tab["robot0_eef_rot_axis_angle"],
                             store.tab["robot0_gripper_width"],
                             store.tab["robot0_demo_start_pose"])
    obs_idx = np.clip(np.array([idx - DOWN_SAMPLE, idx]), s, e - 1)
    T_obs = pose6_to_mat(pos[obs_idx], rv[obs_idx])
    base_inv = np.linalg.inv(T_obs[-1])
    rel_obs = base_inv @ T_obs
    d_obs = pf.mat_to_pose10d(rel_obs)                       # (2, 9)
    sp = start[obs_idx[0]].copy()
    if start_noise > 0.0:
        sp += (rng or np.random.default_rng(0)).normal(0.0, start_noise, 6)
    T_start = pose6_to_mat(sp[:3], sp[3:])
    wrt = pf.mat_to_rot6d((np.linalg.inv(T_start) @ T_obs)[:, :3, :3])
    obs = {
        "robot0_eef_pos": d_obs[:, :3].astype(np.float32),
        "robot0_eef_rot_axis_angle": d_obs[:, 3:9].astype(np.float32),
        "robot0_gripper_width": width[obs_idx].reshape(2, 1).astype(np.float32),
        "robot0_eef_rot_axis_angle_wrt_start": wrt.astype(np.float32),
    }
    act_idx = idx + DOWN_SAMPLE * np.arange(ACTION_HORIZON)
    assert act_idx[-1] < e
    T_act = pose6_to_mat(pos[act_idx], rv[act_idx])
    T_rel = base_inv @ T_act                                 # (16, 4, 4)
    w = width[act_idx].reshape(-1, 1)
    if action_repr == ACTION_REPR_POSE10D:
        d_act = pf.mat_to_pose10d(T_rel)                     # (16, 9)
        action = np.concatenate([d_act, w], axis=1)
    elif action_repr == ACTION_REPR_POS_YAW_WIDTH:
        dp, dyaw = pf.encode_pos_yaw(T_rel)                  # (16, 3), (16,)
        action = np.concatenate([dp, dyaw.reshape(-1, 1), w], axis=1)
    elif action_repr == ACTION_REPR_POS_RPY_WIDTH:
        enc = getattr(pf, "encode_pos_rpy", None)
        if enc is None:
            raise RuntimeError("gt_window: this checkout's policy_frames has no "
                               "encode_pos_rpy (the 6-DoF variant's frames "
                               "transcription is not landed) -- cannot build a "
                               f"{ACTION_REPR_POS_RPY_WIDTH} label")
        dp, dyaw, droll, dpitch = enc(T_rel)                 # (16, 3), (16,) x3
        action = np.concatenate([dp, dyaw.reshape(-1, 1), droll.reshape(-1, 1),
                                 dpitch.reshape(-1, 1), w], axis=1)
    else:
        raise ValueError(f"gt_window: unknown action_repr {action_repr!r} "
                         f"(want one of {sorted(ACTION_DIM_BY_REPR)})")
    action = action.astype(np.float32)
    assert action.shape == (ACTION_HORIZON, ACTION_DIM_BY_REPR[action_repr])
    return obs, action, {"obs_idx": obs_idx, "act_idx": act_idx}


def depth_obs(store: TrainingStore, idx: int, s: int) -> np.ndarray:
    """(2, 3, 224, 224) float32 like sampler.py (first-frame padding) +
    umi_dataset (moveaxis, /255)."""
    num_valid = min(OBS_HORIZON, (idx - s) // DOWN_SAMPLE + 1)
    first = idx - (num_valid - 1) * DOWN_SAMPLE
    frames = store.depth_frames(list(range(first, idx + 1, DOWN_SAMPLE)))
    if frames.shape[0] < OBS_HORIZON:
        frames = np.concatenate([np.repeat(frames[:1], OBS_HORIZON - frames.shape[0],
                                           axis=0), frames], axis=0)
    return (np.moveaxis(frames, -1, 1).astype(np.float32) / 255.0)


# =============================================================================
# filter side
# =============================================================================
def policy_block() -> dict:
    """The v2 defaults, with drift against ``geometry.MpcConfig.policy``
    reported when that block exists (it belongs to another agent)."""
    blk = dict(POLICY_DEFAULTS_V2)
    try:
        from rov_gui.control.geometry import MpcConfig
        live = getattr(MpcConfig(), "policy", None)
    except Exception:                                        # noqa: BLE001
        live = None
    if isinstance(live, dict):
        drift = {k: (blk[k], live.get(k)) for k in blk
                 if k in live and live.get(k) != blk[k]}
        if drift:
            print(f"NOTE geometry.MpcConfig.policy differs from the spec "
                  f"defaults used here: {drift}")
    return blk


def make_limits(blk: dict, plan_stream):
    """FilterLimits from the block — only the fields this checkout's
    FilterLimits has (``require_obs_t`` / ``yaw_jump_max_rad`` are being
    added by another agent)."""
    names = {f.name for f in dataclasses.fields(plan_stream.FilterLimits)}
    box = blk.get("workspace_box_ned")
    kw = dict(v_max=blk["v_max_m_s"], a_max=blk["a_max_m_s2"],
              r_max=blk["r_max_rad_s"], anchor_max_m=blk["anchor_max_m"],
              jump_max_m=blk["jump_max_m"], obs_max_age_s=blk["obs_max_age_s"],
              box_ned_min=tuple(box[0]) if box else None,
              box_ned_max=tuple(box[1]) if box else None)
    if "require_obs_t" in names:
        kw["require_obs_t"] = True
    if "clip_ratio_max" in names and blk.get("clip_ratio_max") is not None:
        kw["clip_ratio_max"] = float(blk["clip_ratio_max"])
    if "yaw_jump_max_rad" in names and blk.get("yaw_jump_max_deg") is not None:
        kw["yaw_jump_max_rad"] = math.radians(blk["yaw_jump_max_deg"])
    # 6-DoF variant attitude gates (plan_stream.FilterLimits rp_* fields,
    # 2026-09-26): only when this checkout's FilterLimits has them; a 4-DoF
    # msg never consults them, so the 5-dim gate numbers are unchanged.
    if "rp_reject_rad" in names and blk.get("rp_reject_deg") is not None:
        kw["rp_reject_rad"] = math.radians(blk["rp_reject_deg"])
    if "rp_rate_max" in names and blk.get("pq_max_rad_s") is not None:
        kw["rp_rate_max"] = float(blk["pq_max_rad_s"])
    if "rp_jump_max_rad" in names and blk.get("rp_jump_max_deg") is not None:
        kw["rp_jump_max_rad"] = math.radians(blk["rp_jump_max_deg"])
    return plan_stream.FilterLimits(**{k: v for k, v in kw.items() if k in names})


def peaks(msg, lim) -> dict:
    """The filter's own finite differences (plan_stream.PlanFilter step 5)."""
    p = np.asarray(msg.p_ned, dtype=float)
    yaw = np.unwrap(np.asarray(msg.yaw, dtype=float))
    v = np.diff(p, axis=1) / msg.dt
    speed = np.linalg.norm(v, axis=0)
    v_peak = float(speed.max()) if speed.size else 0.0
    a_peak = float(np.linalg.norm(np.diff(v, axis=1) / msg.dt, axis=0).max()) \
        if p.shape[1] >= 3 else 0.0
    r = np.diff(yaw) / msg.dt
    r_peak = float(np.abs(r).max()) if r.size else 0.0
    need = max(v_peak / lim.v_max if lim.v_max > 0 else 1.0,
               math.sqrt(a_peak / lim.a_max) if lim.a_max > 0 else 1.0,
               r_peak / lim.r_max if lim.r_max > 0 else 1.0)
    out = dict(v_peak=v_peak, a_peak=a_peak, r_peak=r_peak, need=need)
    # 6-DoF variant: an rp-carrying msg (PlanMsg.rp (2, K) absolute roll /
    # pitch) adds the Euler-rate peaks and the rp_rate term of ``need``
    # (the filter's dilation reads the same finite differences); a 4-DoF
    # msg leaves ``need`` and the dict byte-identical to before 2026-09-26.
    rp = getattr(msg, "rp", None)
    if rp is not None:
        rp = np.asarray(rp, dtype=float).reshape(2, -1)
        rate = np.diff(rp, axis=1) / msg.dt if rp.shape[1] >= 2 else np.zeros((2, 0))
        p_peak = float(np.abs(rate[0]).max()) if rate.size else 0.0
        q_peak = float(np.abs(rate[1]).max()) if rate.size else 0.0
        rp_mag = float(np.abs(rp).max()) if rp.size else 0.0
        rp_rate_max = getattr(lim, "rp_rate_max", None)
        if rp_rate_max:
            out["need"] = max(out["need"], max(p_peak, q_peak) / float(rp_rate_max))
        out.update(p_peak=p_peak, q_peak=q_peak, rp_mag=rp_mag)
    return out


def gate_chunks(chunks: Sequence[np.ndarray], blk: dict, knot_dt: float, pf,
                plan_stream, obs_dt: float, label: str) -> dict:
    """Compose + filter every chunk on one knot grid; print + return stats."""
    lim = make_limits(blk, plan_stream)
    filt = plan_stream.PlanFilter(lim)
    counts = {"accept": 0, "clip": 0, "reject": 0}
    reject_gate: Dict[str, int] = {}
    need, jitter, vp, ap, rp, alphas, compose_err = [], [], [], [], [], [], 0
    pq_peak, rp_mag, rp_clipped_n, gt_rp = [], [], 0, []
    zero6 = np.zeros(6)
    # 7-dim chunks (pos_rpy_width) are composed TRACKED (design D12 sub-mode
    # attitude_track=True: T1 clip at rp_max_deg, PlanMsg.rp carried) so the
    # attitude gates see them; the other widths take the pre-variant call.
    rp_kw: dict = {}
    if chunks and np.asarray(chunks[0]).shape[-1] == ACTION_DIM_BY_REPR[ACTION_REPR_POS_RPY_WIDTH]:
        rp_kw = dict(track_rp=True, rp_max_rad=math.radians(float(blk["rp_max_deg"])))
    for i, a in enumerate(chunks):
        try:
            msg, info = pf.compose_plan(
                a, zero6, (0.0, 0.0), T_BT_GATE, obs_dt, knot_dt, 0.0, i, 0.0,
                blk["gripper_width_open_m"], blk["gripper_width_closed_m"], **rp_kw)
        except ValueError:
            compose_err += 1
            continue
        except TypeError as e:
            if not rp_kw:
                raise
            raise RuntimeError("compose_plan does not take track_rp / rp_max_rad "
                               "(the 6-DoF variant's frames change is not landed "
                               f"in this checkout): {e}") from None
        if rp_kw:
            gt_rp.append(np.degrees(np.abs(np.asarray(a, float)[:, 4:6])))
            rp_clipped_n += int(info.get("rp_clipped_n", 0) or 0)
        filt.reset()
        v = filt.evaluate(msg, r_now=(np.zeros(3), 0.0), cur_sample=None, now=0.0)
        counts[v.status] += 1
        if v.status == "reject":
            gate = v.reasons[0].split(":")[0] if v.reasons else "?"
            reject_gate[gate] = reject_gate.get(gate, 0) + 1
        if "dilation" in v.margins:
            alphas.append(float(v.margins["dilation"]))
        pk = peaks(msg, lim)
        need.append(pk["need"])
        vp.append(pk["v_peak"])
        ap.append(pk["a_peak"])
        rp.append(pk["r_peak"])
        jitter.append(float(info["jitter_rms_mm"]))
        if "p_peak" in pk:
            pq_peak.append(max(pk["p_peak"], pk["q_peak"]))
            rp_mag.append(pk["rp_mag"])

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else float("nan")

    n = sum(counts.values())
    print(f"\n[{label}] knot_dt {knot_dt*1e3:.1f} ms — {n} chunks "
          f"({compose_err} failed to compose)")
    print(f"  accept {counts['accept']} ({100*counts['accept']/max(n,1):.1f}%)  "
          f"clip {counts['clip']} ({100*counts['clip']/max(n,1):.1f}%)  "
          f"reject {counts['reject']} ({100*counts['reject']/max(n,1):.1f}%)"
          + (f"  reject by gate: {reject_gate}" if reject_gate else ""))
    print(f"  need (dilation the kinematic gate wants; <=1 passes, "
          f"<={blk.get('clip_ratio_max', plan_stream.CLIP_RATIO_MAX)} clips): p50 {q(need,50):.2f}  "
          f"p90 {q(need,90):.2f}  max {q(need,100):.2f}"
          + (f"  | applied alpha p50 {q(alphas,50):.2f}" if alphas else ""))
    print(f"  jitter_rms_mm (raw 2nd differences): p50 {q(jitter,50):.2f}  "
          f"p90 {q(jitter,90):.2f}")
    print(f"  peak speed m/s: p50 {q(vp,50):.3f}  p90 {q(vp,90):.3f}  "
          f"(cap {lim.v_max})   peak accel m/s^2: p50 {q(ap,50):.3f}  "
          f"p90 {q(ap,90):.3f}  (cap {lim.a_max})   peak yaw rate rad/s: "
          f"p50 {q(rp,50):.3f}  p90 {q(rp,90):.3f}  (cap {lim.r_max})")
    out = dict(counts=counts, reject_gate=reject_gate, need=need,
               jitter=jitter, v_peak=vp, a_peak=ap, r_peak=rp)
    if rp_kw:
        # the attitude numbers (6-DoF variant): what the gates would do to
        # the demonstrations, and the label distribution behind rp_max_deg
        G = np.concatenate(gt_rp, axis=0) if gt_rp else np.zeros((0, 2))
        n_rp_rej = sum(v for k, v in reject_gate.items() if k.startswith("rp_"))
        print(f"  attitude [{ACTION_REPR_POS_RPY_WIDTH}, tracked, T1 clip "
              f"{blk['rp_max_deg']} deg]: rp_* rejects {n_rp_rej} "
              f"({100*n_rp_rej/max(n,1):.1f}% of chunks; gates "
              f"{ {k: v for k, v in reject_gate.items() if k.startswith('rp_')} }), "
              f"T1-clipped knots {rp_clipped_n}; peak |p|,|q| rad/s: p50 "
              f"{q(pq_peak,50):.3f}  p90 {q(pq_peak,90):.3f}  max {q(pq_peak,100):.3f} "
              f"(cap {getattr(lim, 'rp_rate_max', None)}); plan max|rp| deg: p50 "
              f"{math.degrees(q(rp_mag,50)) if rp_mag else float('nan'):.2f}  p90 "
              f"{math.degrees(q(rp_mag,90)) if rp_mag else float('nan'):.2f} "
              f"(reject {blk['rp_reject_deg']} deg)")
        if G.size:
            print(f"  GT |droll| deg: p50 {q(G[:,0],50):.2f}  p90 {q(G[:,0],90):.2f}  "
                  f"p99 {q(G[:,0],99):.2f}  p99.9 {q(G[:,0],99.9):.2f}  max {q(G[:,0],100):.2f}"
                  f"   |dpitch| deg: p50 {q(G[:,1],50):.2f}  p90 {q(G[:,1],90):.2f}  "
                  f"p99 {q(G[:,1],99):.2f}  p99.9 {q(G[:,1],99.9):.2f}  max {q(G[:,1],100):.2f}"
                  f"   (16-knot windows; the artefact behind rp_max_deg -- "
                  f"a p95 of a few degrees means SLAM jitter, stay 5-dim)")
        out.update(rp_reject=n_rp_rej, pq_peak=pq_peak, rp_mag=rp_mag,
                   rp_clipped_n=rp_clipped_n, gt_rp_deg=G)
    return out


# =============================================================================
# modes
# =============================================================================
def _open(a) -> Tuple[TrainingStore, object, object]:
    dec = BloscDecoder(a.blosc_python)
    print(f"blosc backend: {dec.backend}")
    path = a.data
    if not Path(path).exists() and Path(DEFAULT_DATA_DIR).exists():
        path = DEFAULT_DATA_DIR
    t = time.perf_counter()
    store = TrainingStore(path, dec)
    print(f"store: {store.store.kind} {path}: {store.n} frames, "
          f"{len(store.episode_ends)} episodes, fps {store.fps} "
          f"({time.perf_counter()-t:.1f} s to load the tables)")
    from rov_gui.control import policy_frames as pf
    from rov_gui.control import plan_stream
    return store, pf, plan_stream


def mode_gate(a) -> int:
    store, pf, plan_stream = _open(a)
    blk = policy_block()
    obs_dt = DOWN_SAMPLE / store.fps
    repr_ = a.action_repr or POLICY_ACTION_REPR
    wins = window_indices(store, a.stride, a.windows)
    print(f"windows: {len(wins)} (stride {a.stride} frames, >= {a.windows} "
          f"wanted), obs_dt {obs_dt*1e3:.1f} ms, action_repr {repr_} "
          f"({ACTION_DIM_BY_REPR[repr_]} columns; the two reprs' yaw gates read "
          f"different definitions — not comparable, see the module docstring), "
          f"T_bt = axis permutation, zero lever arm; limits "
          f"{json.dumps({k: blk[k] for k in blk if k != 'workspace_box_ned'})}"
          f" box {blk['workspace_box_ned']}")
    chunks = [gt_window(store, i, s, e, pf, action_repr=repr_)[1]
              for i, s, e in wins]
    for kdt in a.knot_dt:
        gate_chunks(chunks, blk, kdt, pf, plan_stream, obs_dt, f"GT demos {repr_}")
    return 0


def _session(a):
    """(session, action_repr) — or (None, None) when this mode cannot run.

    The representation is the checkpoint CONTRACT's (``action_repr`` when the
    loader publishes it, else the width: 5 -> pos_yaw_width, 10 -> pose10d);
    ``--action-repr`` is only a cross-check here, never a choice — a flag
    that disagrees with the checkpoint is a mistake worth stopping for.
    """
    from rov_gui.perception import dp_policy
    try:
        import torch
    except ImportError:
        print("torch not importable — this mode needs the GPU session; skipping")
        return None, None
    if not torch.cuda.is_available():
        print("CUDA unavailable — this mode needs the GPU session; skipping")
        return None, None
    if not Path(a.ckpt).is_file():
        print(f"checkpoint missing: {a.ckpt} — skipping")
        return None, None
    s = dp_policy.DpPolicySession(a.ckpt, a.repo, num_inference_steps=a.steps,
                                  eval_transforms=a.eval_transforms,
                                  weights=a.weights)
    s.load(lambda m: print("  " + m))
    print(f"  weights: {a.weights}  (the station's --policy-weights; runs on "
          f"different weights must not be pooled)")
    c = s.contract
    assert c["action_horizon"] == ACTION_HORIZON and c["obs_horizon"] == OBS_HORIZON
    dim = int(c["action_dim"])
    repr_ = c.get("action_repr") or ACTION_REPR_BY_DIM.get(dim)
    assert repr_ in ACTION_DIM_BY_REPR and ACTION_DIM_BY_REPR[repr_] == dim, (
        f"checkpoint contract action_dim {dim} / action_repr "
        f"{c.get('action_repr')!r} is none of (5, {ACTION_REPR_POS_YAW_WIDTH}), "
        f"(7, {ACTION_REPR_POS_RPY_WIDTH}), (10, {ACTION_REPR_POSE10D})")
    assert a.action_repr in (None, repr_), (
        f"--action-repr {a.action_repr} but the checkpoint contract is {repr_} "
        f"(action_dim {dim}): the tool follows the checkpoint — drop the flag or "
        f"point --ckpt at the intended run")
    print(f"  action contract: {repr_} ({dim} columns)"
          + (f", rpy_convention {c.get('rpy_convention')}"
             if repr_ == ACTION_REPR_POS_RPY_WIDTH else ""))
    return s, repr_


def rot_err_deg(d6_a: np.ndarray, d6_b: np.ndarray, pf) -> np.ndarray:
    Ra = pf.rot6d_to_mat(d6_a)
    Rb = pf.rot6d_to_mat(d6_b)
    Rd = np.swapaxes(Ra, -1, -2) @ Rb
    tr = np.clip((np.trace(Rd, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def wrap_pi(x) -> np.ndarray:
    """Elementwise wrap to (-pi, pi]."""
    x = np.asarray(x, dtype=float)
    return np.arctan2(np.sin(x), np.cos(x))


#: Which column carries ``dyaw`` per representation (design D1: col 3 for
#: BOTH flyable reprs, so a 7-dim checkpoint's cols [:, :4] compare directly
#: with the 5-dim baselines); pose10d has no yaw column (derived instead).
DYAW_COL = {ACTION_REPR_POS_YAW_WIDTH: 3, ACTION_REPR_POS_RPY_WIDTH: 3}
#: The (roll, pitch) columns of the 7-dim repr; None elsewhere.
RP_COLS = {ACTION_REPR_POS_RPY_WIDTH: (4, 5)}


def rp_of(chunk: np.ndarray, repr_: str) -> Optional[np.ndarray]:
    """(K, 2) [droll, dpitch] rad of a chunk, None for a repr without the
    columns (pos_yaw_width, pose10d)."""
    cols = RP_COLS.get(repr_)
    if cols is None:
        return None
    return np.asarray(chunk[:, list(cols)], dtype=float)


def ang_err_deg(act: np.ndarray, gt: np.ndarray, repr_: str, pf) -> np.ndarray:
    """Per-knot angular error [deg]: the wrapped ``dyaw`` difference of
    column 3 (pos_yaw_width / pos_rpy_width) or the geodesic rot6d error
    (pose10d)."""
    if repr_ == ACTION_REPR_POSE10D:
        return rot_err_deg(act[:, 3:9], gt[:, 3:9], pf)
    j = DYAW_COL[repr_]
    return np.degrees(wrap_pi(act[:, j] - gt[:, j]))


def dyaw_of(chunk: np.ndarray, repr_: str, pf) -> np.ndarray:
    """(K,) mount-referenced dyaw of a chunk: column 3 for pos_yaw_width and
    pos_rpy_width (DYAW_COL); for pose10d DERIVED from the rot6d through the
    same encoder (``encode_pos_yaw(pose10d_to_mat(.))``) so the old and the
    new checkpoint read on one yaw scale."""
    if repr_ == ACTION_REPR_POSE10D:
        return np.asarray(pf.encode_pos_yaw(pf.pose10d_to_mat(chunk[:, :9]))[1],
                          dtype=float)
    return np.asarray(chunk[:, DYAW_COL[repr_]], dtype=float)


def action_range(s) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The contract's per-column output bound [min, max] = the action
    normalizer's ``input_stats`` (DDIM ``clip_sample`` keeps every
    range-normalised column inside it): ``contract['action_range']`` when the
    loader publishes it, else straight from the numpy normalizer."""
    r = s.contract.get("action_range")
    if r is not None:
        if isinstance(r, dict):
            return np.asarray(r["min"], float), np.asarray(r["max"], float)
        return np.asarray(r[0], float), np.asarray(r[1], float)
    n = s.normalizer
    if n is not None and "action" in n.keys:
        st = n.input_stats("action")
        if "min" in st and "max" in st:
            return np.asarray(st["min"], float), np.asarray(st["max"], float)
    return None


RANGE_TOL = 1e-4


def check_action_range(acts: np.ndarray, s, label: str) -> None:
    """Count, print and ASSERT: every predicted value of a range-normalised
    column lies inside the contract's action_range (+-RANGE_TOL). Identity-
    normalised columns (the legacy checkpoint's rot6d, scale 1 / offset 0)
    are not range-bound by clip_sample, so they are reported, not asserted."""
    rng = action_range(s)
    x = np.asarray(acts, dtype=float).reshape(-1, np.asarray(acts).shape[-1])
    if rng is None:
        print(f"  {label}: action_range unavailable (no input_stats in the "
              f"normalizer) — range check skipped")
        return
    lo, hi = rng
    viol = ((x < lo - RANGE_TOL) | (x > hi + RANGE_TOL)).sum(axis=0)
    scale, offset = s.normalizer.params("action")
    ident = np.isclose(np.asarray(scale, float), 1.0) & np.isclose(np.asarray(offset, float), 0.0)
    print(f"  {label}: values outside the contract action_range +-{RANGE_TOL:g}, "
          f"per column: {viol.tolist()} of {x.shape[0]}; range-normalised "
          f"columns {np.flatnonzero(~ident).tolist()}"
          + (f", identity-normalised {np.flatnonzero(ident).tolist()} reported only"
             if ident.any() else "")
          + f"; range min {np.round(lo, 4).tolist()} max {np.round(hi, 4).tolist()}")
    bad = int(viol[~ident].sum())
    assert bad == 0, (
        f"{label}: {bad} predicted values outside the contract action_range "
        f"(+-{RANGE_TOL:g}) in range-normalised columns "
        f"{np.flatnonzero((viol > 0) & ~ident).tolist()} — the DDIM clip_sample "
        f"bound does not hold for this checkpoint / loader")


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = math.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    return float(np.sum(x * y) / d) if d > 0.0 else float("nan")


def heldout_report(pol_chunks, gt_chunks, repr_, pf, obs_dt: float, w_col: int
                   ) -> dict:
    """The acceptance numbers (module docstring, replay --val-only): per-
    element MSE like the upstream ``val_action_mse_error_*`` (mean over
    windows x knots x coordinates, RAW units), per knot and overall."""
    P = np.asarray(pol_chunks, dtype=float)                  # (N, 16, D)
    G = np.asarray(gt_chunks, dtype=float)
    err = P - G
    n = P.shape[0]
    pos_mse_k = np.mean(err[:, :, :3] ** 2, axis=(0, 2))    # (16,)
    wid_mse_k = np.mean(err[:, :, w_col] ** 2, axis=0)
    dy_p = np.array([dyaw_of(c, repr_, pf) for c in pol_chunks])   # (N, 16)
    dy_g = np.array([dyaw_of(c, repr_, pf) for c in gt_chunks])
    dy_e = wrap_pi(dy_p - dy_g)
    yaw_rms_k = np.degrees(np.sqrt(np.mean(dy_e ** 2, axis=0)))
    yaw_zero_k = np.degrees(np.sqrt(np.mean(dy_g ** 2, axis=0)))
    corr_k = [_corr(dy_p[:, k], dy_g[:, k]) for k in range(P.shape[1])]
    derived = repr_ == ACTION_REPR_POSE10D
    print(f"\nheld-out report over {n} windows — de-normalised, per-element MSE "
          f"(mean over windows x knots x coords, the upstream val_action_mse_error_* "
          f"definition; RMSE = sqrt):")
    print(f"  yaw = {'column 3 (dyaw)' if not derived else 'DERIVED dyaw = encode_pos_yaw(pose10d_to_mat(rot6d))'}; "
          f"yaw_zero = the zero-predictor baseline sqrt(mean(gt_dyaw^2)); corr = corr(pred dyaw, gt dyaw) "
          f"(nan where GT dyaw is constant, e.g. knot 0 == 0 by construction)")
    print("  k   t_ms   pos_mse_m2  pos_rmse_mm  yaw_rms_deg  yaw_zero_deg  yaw_corr  width_rmse_mm"
          + ("  rot_rms_deg" if derived else ""))
    if derived:
        rot_k = np.sqrt(np.mean(np.array([rot_err_deg(p[:, 3:9], g[:, 3:9], pf)
                                          for p, g in zip(pol_chunks, gt_chunks)]) ** 2, axis=0))
    for k in range(P.shape[1]):
        print(f"  {k:2d}  {k*obs_dt*1e3:6.1f}  {pos_mse_k[k]:.6f}  {1e3*math.sqrt(pos_mse_k[k]):11.1f}  "
              f"{yaw_rms_k[k]:11.2f}  {yaw_zero_k[k]:12.2f}  {corr_k[k]:8.3f}  "
              f"{1e3*math.sqrt(wid_mse_k[k]):13.1f}"
              + (f"  {rot_k[k]:11.2f}" if derived else ""))
    pos_mse = float(np.mean(err[:, :, :3] ** 2))
    wid_mse = float(np.mean(err[:, :, w_col] ** 2))
    yaw_rms = math.degrees(math.sqrt(float(np.mean(dy_e ** 2))))
    yaw_zero = math.degrees(math.sqrt(float(np.mean(dy_g ** 2))))
    corr = _corr(dy_p.ravel(), dy_g.ravel())
    total = float(np.mean(err ** 2))
    out = dict(n=n, pos_mse_m2=pos_mse, pos_rmse_mm=1e3 * math.sqrt(pos_mse),
               yaw_rms_deg=yaw_rms, yaw_zero_deg=yaw_zero, yaw_corr=corr,
               width_rmse_mm=1e3 * math.sqrt(wid_mse), action_mse=total)
    print(f"  ALL: pos MSE {pos_mse:.6f} m^2 = RMSE {1e3*math.sqrt(pos_mse):.1f} mm | "
          f"yaw RMS {yaw_rms:.2f} deg vs zero-predictor {yaw_zero:.2f} deg "
          f"(ratio {yaw_rms/yaw_zero if yaw_zero > 0 else float('nan'):.2f}), "
          f"corr(pred, gt) {corr:+.3f}"
          + (f" | rot RMS {math.sqrt(float(np.mean(rot_k**2))):.2f} deg" if derived else "")
          + f" | width RMSE {1e3*math.sqrt(wid_mse):.1f} mm | action MSE all "
          f"{P.shape[2]} columns {total:.6f} (NOT comparable across reprs)")
    if repr_ == ACTION_REPR_POS_RPY_WIDTH:
        out.update(heldout_rp_report(pol_chunks, gt_chunks, repr_, obs_dt))
    return out


def heldout_rp_report(pol_chunks, gt_chunks, repr_, obs_dt: float) -> dict:
    """The 6-DoF variant's held-out ACCEPTANCE rows (training_plan 6,
    2026-09-26), printed AFTER the pos / yaw / width table so the 5-dim rows
    above stay byte-identical: per knot and overall, roll / pitch RMS [deg]
    beside the zero-predictor baseline and corr(pred, GT) per axis, and the
    GT |droll| / |dpitch| percentiles (the artefact behind ``rp_max_deg``).
    Gate [예측]: roll / pitch RMS below the zero-predictor and corr > 0.5."""
    Rp = np.array([rp_of(c, repr_) for c in pol_chunks])          # (N, 16, 2)
    Rg = np.array([rp_of(c, repr_) for c in gt_chunks])
    e = wrap_pi(Rp - Rg)
    rms_k = np.degrees(np.sqrt(np.mean(e ** 2, axis=0)))         # (16, 2)
    zero_k = np.degrees(np.sqrt(np.mean(Rg ** 2, axis=0)))
    K = Rp.shape[1]
    corr_k = np.array([[_corr(Rp[:, k, j], Rg[:, k, j]) for j in range(2)]
                       for k in range(K)])
    print(f"\n  attitude columns ({repr_}, ZYX roll / pitch of R_bt R_rel R_bt^T, deg; "
          f"*_zero = zero-predictor sqrt(mean(gt^2)); corr nan where GT is constant):")
    print("  k   t_ms   roll_rms  roll_zero  roll_corr  pitch_rms  pitch_zero  pitch_corr")
    for k in range(K):
        print(f"  {k:2d}  {k*obs_dt*1e3:6.1f}  {rms_k[k,0]:8.2f}  {zero_k[k,0]:9.2f}  "
              f"{corr_k[k,0]:9.3f}  {rms_k[k,1]:9.2f}  {zero_k[k,1]:10.2f}  {corr_k[k,1]:10.3f}")
    roll_rms = math.degrees(math.sqrt(float(np.mean(e[:, :, 0] ** 2))))
    pitch_rms = math.degrees(math.sqrt(float(np.mean(e[:, :, 1] ** 2))))
    roll_zero = math.degrees(math.sqrt(float(np.mean(Rg[:, :, 0] ** 2))))
    pitch_zero = math.degrees(math.sqrt(float(np.mean(Rg[:, :, 1] ** 2))))
    roll_corr = _corr(Rp[:, :, 0].ravel(), Rg[:, :, 0].ravel())
    pitch_corr = _corr(Rp[:, :, 1].ravel(), Rg[:, :, 1].ravel())
    G = np.degrees(np.abs(Rg)).reshape(-1, 2)

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else float("nan")

    print(f"  ALL: roll RMS {roll_rms:.2f} deg vs zero {roll_zero:.2f} "
          f"(ratio {roll_rms/roll_zero if roll_zero > 0 else float('nan'):.2f}), corr {roll_corr:+.3f} | "
          f"pitch RMS {pitch_rms:.2f} deg vs zero {pitch_zero:.2f} "
          f"(ratio {pitch_rms/pitch_zero if pitch_zero > 0 else float('nan'):.2f}), corr {pitch_corr:+.3f}")
    print(f"  GT |droll| deg: p50 {q(G[:,0],50):.2f}  p90 {q(G[:,0],90):.2f}  p99 {q(G[:,0],99):.2f}  "
          f"p99.9 {q(G[:,0],99.9):.2f}   |dpitch| deg: p50 {q(G[:,1],50):.2f}  p90 {q(G[:,1],90):.2f}  "
          f"p99 {q(G[:,1],99):.2f}  p99.9 {q(G[:,1],99.9):.2f}   (held-out windows)")
    return dict(roll_rms_deg=roll_rms, pitch_rms_deg=pitch_rms,
                roll_zero_deg=roll_zero, pitch_zero_deg=pitch_zero,
                roll_corr=roll_corr, pitch_corr=pitch_corr,
                gt_droll_p99_deg=q(G[:, 0], 99), gt_dpitch_p99_deg=q(G[:, 1], 99))


def mode_replay(a) -> int:
    store, pf, plan_stream = _open(a)
    s, repr_ = _session(a)
    if s is None:
        return 0
    blk = policy_block()
    obs_dt = DOWN_SAMPLE / store.fps
    if abs(obs_dt - s.contract["obs_dt_s"]) > 1e-9:
        print(f"obs_dt mismatch: store {obs_dt} vs session {s.contract['obs_dt_s']}")
        return 1
    w_col = ACTION_DIM_BY_REPR[repr_] - 1
    if a.val_only:
        sel, eps = val_windows(store, a.val_stride)
        starts = {store.episodes()[k][0]: k for k in eps}
        per_ep = {k: 0 for k in eps}
        for _i, s0, _e in sel:
            per_ep[starts[s0]] += 1
        print(f"replay --val-only: {len(sel)} HELD-OUT windows — episodes {eps} by "
              f"get_val_mask({len(store.episode_ends)}, {VAL_RATIO}, {VAL_SEED}), "
              f"per episode {per_ep}, every valid start at stride {a.val_stride}"
              + (" (= the 219-window set of KNOWN_ISSUES.md:33-46)"
                 if a.val_stride == 4 and len(sel) == 219 else "")
              + f"; eval_transforms {a.eval_transforms}, {a.steps} DDIM steps, "
              f"seed per window = frame index")
    else:
        wins = window_indices(store, a.stride, max(a.windows, 8 * a.replay_windows))
        sel = [wins[i] for i in np.linspace(0, len(wins) - 1, a.replay_windows).astype(int)]
        print(f"replay: {len(sel)} windows of {len(wins)} (whole store, mostly "
              f"TRAIN; --val-only for the held-out split), eval_transforms "
              f"{a.eval_transforms}, seed per window = frame index")
    # prefetch every depth frame in one decoder call
    need = sorted({int(j) for i, s0, e in sel
                   for j in range(max(s0, i - DOWN_SAMPLE), i + 1, DOWN_SAMPLE)})
    t = time.perf_counter()
    store.depth.prefetch(need)
    print(f"  {len(need)} depth frames decoded in {time.perf_counter()-t:.1f} s")
    pos_e, ang_e, wid_e, ms, gt_chunks, pol_chunks = [], [], [], [], [], []
    for i, s0, e in sel:
        low, gt, _ = gt_window(store, i, s0, e, pf, action_repr=repr_)
        obs = dict(low)
        obs["camera0_depth"] = depth_obs(store, i, s0)
        act, m = s.predict(obs, seed=int(i))
        assert act.shape == gt.shape, (act.shape, gt.shape)
        ms.append(m)
        pos_e.append(np.linalg.norm(act[:, :3] - gt[:, :3], axis=1))
        ang_e.append(ang_err_deg(act, gt, repr_, pf))
        wid_e.append(np.abs(act[:, w_col] - gt[:, w_col]))
        gt_chunks.append(gt)
        pol_chunks.append(act)
    pos_e, ang_e, wid_e = np.array(pos_e), np.array(ang_e), np.array(wid_e)
    rms = lambda x: np.sqrt(np.mean(x * x, axis=0))               # noqa: E731
    ang_hdr = "yaw_rms_deg" if repr_ != ACTION_REPR_POSE10D else "rot_rms_deg"
    ang_lbl = "yaw" if repr_ != ACTION_REPR_POSE10D else "rot"
    print(f"\nreplay vs ground truth over {len(sel)} windows [{repr_}] "
          f"(RMS across windows, per knot index; pos = |vector| RMS):")
    print(f"  k   t_ms   pos_rms_mm  {ang_hdr}  width_rms_mm")
    for k in range(ACTION_HORIZON):
        print(f"  {k:2d}  {k*obs_dt*1e3:6.1f}  {1e3*rms(pos_e)[k]:9.1f}  "
              f"{rms(ang_e)[k]:10.2f}  {1e3*rms(wid_e)[k]:11.1f}")
    print(f"  all knots: pos RMS {1e3*math.sqrt(np.mean(pos_e**2)):.1f} mm, "
          f"{ang_lbl} RMS {math.sqrt(np.mean(ang_e**2)):.2f} deg, width RMS "
          f"{1e3*math.sqrt(np.mean(wid_e**2)):.1f} mm; GT chunk end |pos| p50 "
          f"{1e3*np.median([np.linalg.norm(g[-1,:3]) for g in gt_chunks]):.0f} mm")
    print(f"  infer_ms: p50 {np.percentile(ms,50):.1f}  max {max(ms):.1f}  "
          f"(n {len(ms)}, {a.steps} DDIM steps)")
    if a.val_only:
        heldout_report(pol_chunks, gt_chunks, repr_, pf, obs_dt, w_col)
    # after the numbers, so a broken bound never hides them
    check_action_range(np.array(pol_chunks), s, "policy chunks")
    for kdt in a.knot_dt:
        gate_chunks(pol_chunks, blk, kdt, pf, plan_stream, obs_dt, "POLICY chunks")
        gate_chunks(gt_chunks, blk, kdt, pf, plan_stream, obs_dt,
                    "GT chunks (same windows)")
    s.close()
    return 0


def mode_noise(a) -> int:
    store, pf, plan_stream = _open(a)
    s, repr_ = _session(a)
    if s is None:
        return 0
    w_col = ACTION_DIM_BY_REPR[repr_] - 1
    wins = window_indices(store, a.stride, max(a.windows, 8 * a.noise_windows))
    sel = [wins[i] for i in np.linspace(0, len(wins) - 1, a.noise_windows).astype(int)]
    sigmas = [float(x) for x in a.sigma_mm]
    print(f"noise: {len(sel)} windows [{repr_}], sigma_mm {sigmas} on the OLDER "
          f"relative-pose row (the newest row is identically zero by construction), "
          f"diffusion noise fixed per window (seed = frame index)")
    need = sorted({int(j) for i, s0, e in sel
                   for j in range(max(s0, i - DOWN_SAMPLE), i + 1, DOWN_SAMPLE)})
    store.depth.prefetch(need)
    rng = np.random.default_rng(a.seed)
    per_sigma = {sg: [] for sg in sigmas}
    gts = []
    for i, s0, e in sel:
        low, gt, _ = gt_window(store, i, s0, e, pf, action_repr=repr_)
        d = depth_obs(store, i, s0)
        gts.append(gt)
        noise = rng.normal(0.0, 1.0, 3)
        for sg in sigmas:
            obs = {k: v.copy() for k, v in low.items()}
            obs["camera0_depth"] = d
            obs["robot0_eef_pos"][0] += (sg * 1e-3 * noise).astype(np.float32)
            act, _ = s.predict(obs, seed=int(i))
            per_sigma[sg].append(act)
    ref = np.array(per_sigma[sigmas[0]])
    gts = np.array(gts)
    ang_hdr = "d_yaw_rms_deg" if repr_ != ACTION_REPR_POSE10D else "d_rot_rms_deg"
    print(f"\n  sigma_mm  d_pos_rms_mm(vs sigma0)  d_pos_max_mm  {ang_hdr}  "
          f"d_width_rms_mm | pos_rms_vs_GT_mm")
    for sg in sigmas:
        acts = np.array(per_sigma[sg])
        dp = np.linalg.norm(acts[:, :, :3] - ref[:, :, :3], axis=-1)
        if repr_ == ACTION_REPR_POSE10D:
            dr = rot_err_deg(acts[:, :, 3:9].reshape(-1, 6),
                             ref[:, :, 3:9].reshape(-1, 6), pf)
        else:
            j = DYAW_COL[repr_]
            dr = np.degrees(wrap_pi(acts[:, :, j] - ref[:, :, j]))
        dw = acts[:, :, w_col] - ref[:, :, w_col]
        pg = np.linalg.norm(acts[:, :, :3] - gts[:, :, :3], axis=-1)
        print(f"  {sg:7.1f}  {1e3*math.sqrt(np.mean(dp**2)):22.2f}  "
              f"{1e3*dp.max():12.1f}  {math.sqrt(np.mean(dr**2)):13.3f}  "
              f"{1e3*math.sqrt(np.mean(dw**2)):14.2f} | "
              f"{1e3*math.sqrt(np.mean(pg**2)):.1f}")
    print(f"  (training input range of the older row, from the normalizer: "
          f"{s.describe()['normalizer_input_stats'].get('robot0_eef_pos')})")
    check_action_range(np.concatenate([np.array(v) for v in per_sigma.values()]),
                       s, "emitted chunks (all sigmas)")
    s.close()
    return 0


def main(argv=None) -> int:
    from rov_gui.perception import dp_policy
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("gate", "replay", "noise"))
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="zarr zip or directory (default: the training zip)")
    ap.add_argument("--blosc-python", default=None,
                    help="helper interpreter with numcodecs (else auto)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--windows", type=int, default=500, help="gate: minimum windows")
    ap.add_argument("--replay-windows", type=int, default=100)
    ap.add_argument("--noise-windows", type=int, default=50)
    ap.add_argument("--sigma-mm", nargs="+", default=[0.0, 2.0, 5.0, 10.0])
    ap.add_argument("--knot-dt", nargs="+", type=float, default=list(KNOT_DTS))
    ap.add_argument("--action-repr", default=None,
                    choices=(ACTION_REPR_POS_YAW_WIDTH, ACTION_REPR_POS_RPY_WIDTH,
                             ACTION_REPR_POSE10D),
                    help="gate: the GT label representation (default "
                         f"{POLICY_ACTION_REPR}, the flown contract; pos_rpy_width = "
                         "the 6-DoF variant's 16x7 labels through the attitude "
                         "gates; pose10d = the legacy 16x10 gate). replay/noise: "
                         "taken from the checkpoint contract — when given it must "
                         "agree")
    ap.add_argument("--val-only", action="store_true",
                    help="replay: the HELD-OUT episodes of the training split "
                         "(get_val_mask verbatim) instead of the stride-8 "
                         "whole-store selection; prints the acceptance numbers")
    ap.add_argument("--val-stride", type=int, default=4,
                    help="replay --val-only: stride over the valid starts of each "
                         "held-out episode; 4 = the 219-window set of "
                         "KNOWN_ISSUES.md:33-46, 1 = every valid start (869)")
    ap.add_argument("--ckpt", default=dp_policy.DEFAULT_CKPT)
    ap.add_argument("--repo", default=dp_policy.DEFAULT_UMI_REPO)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--weights", default="ema_model",
                    help="which state dict inside the checkpoint to run: "
                         "ema_model (deployed default) or model (the non-EMA "
                         "workaround for the BatchNorm defect, KNOWN_ISSUES.md); "
                         "the station flag is --policy-weights")
    ap.add_argument("--eval-transforms", default="center",
                    choices=dp_policy.EVAL_TRANSFORMS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return {"gate": mode_gate, "replay": mode_replay, "noise": mode_noise}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
