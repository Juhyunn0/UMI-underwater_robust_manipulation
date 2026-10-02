#!/usr/bin/env python3
"""
fstereo.py — FoundationStereo (NVlabs) as a depth source for the station.

Why this exists
---------------
The C3's on-board stereo depth is not metric in water. It reads 1.56x long
against a caliper and 1.71x against the tag map, and the error GROWS WITH
RANGE, so the ``--depth-scale`` stopgap is exact at one distance only
(KNOWN_ISSUES 2026-08-24). A learned matcher produces its own disparity from
the same rectified pair, so it is the one experiment that can answer "is the
geometry recoverable at all, or is this camera's stereo pair mis-calibrated
beyond a scalar" — by putting a second depth map on screen next to the first.

This module is the model half of that. It owns the network and nothing else:
no camera, no Qt, no mailbox. The worker in ``backends/hardware.py`` feeds it
raw mono pairs and publishes what comes back.

The pipeline, and where each piece comes from
---------------------------------------------
    raw left/right (640x400 gray8, straight off the OV9282 pair)
      -> rectify            c3_camera.host_depth  StereoRig.map_left/map_right
      -> disparity (px)     FoundationStereo      this module
      -> uint16 mm          c3_camera.host_depth  depth_from_disparity
      -> colour grid        c3_camera.host_depth  warp_depth_to_color
      -> panel grid         c3_camera.host_depth  resize_depth_nearest

Only the second line is new. ``host_depth.py`` already implements the rest and
was, until now, dead code at runtime — reusing it means the learned depth map
carries byte-for-byte the same contract as the device's (uint16 millimetres,
0 = no measurement) and therefore drops into the existing panel, colour bar,
cursor probe and depth-vs-MAP check without any of them knowing.

Why the RAW pair and not the device's ``rectifiedLeft/Right``
-------------------------------------------------------------
The reference tool this is ported from
(``~/Desktop/data collection/UMI_Underwater/oakd_foundation_stereo.py``) takes the
device's rectified pair and converts disparity with
``getCameraIntrinsics(CAM_C)[0][0]``. That is the UNRECTIFIED focal length.
DepthAI 2.32 exposes no API that returns a rectified one — only the 3x3
rectification rotations — so pairing device-rectified images with an EEPROM fx
is a ~0.4% systematic scale error: far too small to look broken, exactly large
enough to poison a map (``host_depth.py:43-61`` refuses it by construction).

Rectifying here instead makes ``P1[0,0]`` the focal length of the very images
being matched, by definition. Cost: 0.21-0.36 ms for the 640x400 pair
[유도: host_depth.py:1256 comment, not re-measured here]. It is the only
deliberate deviation from the reference setting; the model side (checkpoint,
iters, scale, tensor prep, fp16 autocast) is carried over unchanged — except
that the autocast weight cache is off in both the graph and the eager path (a
capture requirement; every number in rov_gui/tools/fstereo_bench_out includes
it).

Import-time hazards, all three of them real
--------------------------------------------
1. ``XFORMERS_DISABLED`` must be set BEFORE torch/dinov2 are imported — dinov2
   reads it at module scope in three files, so setting it afterwards is too
   late. The pinned xformers build in FoundationStereo's own environment.yml
   has no Blackwell kernels, and this rig is an RTX 5090 (sm_120).
2. FoundationStereo does bare ``from core...`` / ``from Utils import ...``, so
   its checkout must be on ``sys.path``. ``Utils`` is a TOP-LEVEL name — and
   FoundationPose (``--pose``, same process) has one too. See
   :func:`_import_upstream`: the name is owned only for the duration of the
   import and the checkout is then moved to the END of ``sys.path``.
3. ``FoundationStereo/Utils.py`` calls ``set_logging_format()`` at module
   scope, which does ``importlib.reload(logging)`` + ``basicConfig``. In a
   library that is rude; in a GUI process that also flies a vehicle it would
   silently reconfigure the station's logging. The import is therefore wrapped
   in :func:`_preserve_logging`.

Grad is off per-call, not globally: ``torch.autograd.set_grad_enabled`` is
THREAD-LOCAL, and inference runs on the worker thread, not the loader thread
that built the model.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

#: Where the checkpoint lives, relative to the FoundationStereo checkout.
#: 23-51-11 is the ViT-large model (3.07 GiB). The ViT-small alternative the
#: upstream readme mentions (11-33-40) sits beside it; nothing here uses it.
DEFAULT_CKPT = Path("pretrained_models") / "23-51-11" / "model_best_bp2.pth"

#: SHA-1 of the first 8 MiB (:func:`_sha1_head`) of upstream's two released
#: checkpoints, by model id ``<run>/<file>``: the reference the FS parity check
#: (backends/policy.py ``same_fstereo_ckpt``) needs when a training store names
#: a checkpoint file that no longer exists — the stores built before 2026-09-27
#: name the deleted ~/Desktop/FoundationStereo copy. Computed on
#: external/FoundationStereo/pretrained_models/*/model_best_bp2.pth on
#: 2026-09-30; 23-51-11's is also the ckpt_sha1_first_8mib of every run before.
UPSTREAM_CKPT_SHA1 = {
    "23-51-11/model_best_bp2.pth": "d277c7ff7de5c627328cde9eaf775e181685276a",
    "11-33-40/model_best_bp2.pth": "a2445fc23818abd52f8eedda3c3bf3b6b2bdc496",
}

#: The FoundationStereo checkout: the git submodule external/FoundationStereo
#: (NVlabs upstream, pinned at 6e88068). The weights are NOT in git — upstream's
#: own .gitignore drops pretrained_models/ — so a fresh clone of this repo has
#: the code but must have them copied in (rov_gui/README.md, section
#: "FoundationStereo 체크아웃"). Until 2026-09-30 this was ~/Desktop/FoundationStereo/
#: FoundationStereo, OUTSIDE the repo, and was deleted as unused for exactly
#: that reason; every run recorded before then names the old path in its meta.
DEFAULT_REPO = Path(__file__).resolve().parents[2] / "external" / "FoundationStereo"

#: GRU refinement iterations. 32 is the paper default and the reference tool
#: uses 16; at the station's scale 0.5 they are indistinguishable from 8 on
#: real frames (median disagreement with the reference 0.165 vs 0.161 px) and
#: 8 is 15 ms cheaper (44.2 vs 58.8 ms eager) [측정 2026-09-02,
#: rov_gui/tools/fstereo_bench_out/sweep.txt]. See --fstereo-iters.
DEFAULT_ITERS = 8

#: NO RANGE CUT. Only a non-positive disparity is "no measurement"; every
#: other pixel keeps the distance the network predicted, up to the uint16
#: ceiling. This matches the reference viewer, and it is why the reference
#: showed a solid map while the station's first hardware run showed 18% black:
#: a (200, 15000) mm cut here was zeroing the gripper 15 cm in front of the
#: lens and the far tent wall, and nothing else. On nine real stereo frames
#: the network returns a positive disparity on 100.00% of pixels, and the
#: projection onto the colour grid loses 0.36% at the border [측정
#: 2026-09-02, rov_gui/tools/fstereo_bench_out/session.txt, columns native %
#: / loss %] — so the cut was the whole loss. Anything nearer than the colour
#: bar's 300 mm simply paints
#: as bottom-of-scale, which is what the device's own depth does too.
Z_RANGE_MM = (0.0, 65535.0)

#: Replay the network as ONE CUDA graph instead of ~6,500 separate kernel
#: launches. At the station's input size the forward is launch-bound, not
#: compute-bound: 6,572 kernels in 44.2 ms is 6.7 us per kernel [유도] — the
#: CPU cost of launching one — and shrinking the input from 320 to 160 px
#: wide only takes 44.2 to 37.1 ms. Captured once, the same forward replays
#: in 36.8 ms (45.4 eager) with identical output (np.array_equal on all nine
#: pairs at ten settings), and a replay is a single call, so it is also
#: immune to the GIL contention that turned ~44 ms into 81 ms inside the
#: running station (2026-09-02 09:05 panel: 11.4 Hz / 81 ms, GPU otherwise
#: idle). [측정 2026-09-02, RTX 5090 유휴, 9 real 640x400 pairs:
#: rov_gui/tools/fstereo_bench_out/sweep.txt, graph.txt]. Capture failing
#: falls back to eager, once, and says so (see FStereoSession._capture).
DEFAULT_GRAPH = True

#: Iterations of :func:`c3_camera.host_depth.fill_scatter_gaps` on the warped
#: map. The projection onto the colour grid leaves a web of 1-3 px black curves
#: along every depth gradient — 4.68% of the grid on real C3 frames, in 539
#: connected components that are all thin lines. It is a resampling artifact of
#: the forward scatter, not missing information, and it is the whole reason the
#: station's depth panel looked speckled next to the reference viewer's (which
#: never leaves the mono camera's own frame, so it cannot have these). 2 passes
#: reach 100.00% fill for 0.14 ms and change ZERO measured pixels [측정
#: 2026-09-02, rov_gui/tools/fstereo_bench_out/hardware_20260902_fill.txt].
#: 0 turns it off and the panel shows the raw projection.
DEFAULT_FILL = 2


class FStereoError(RuntimeError):
    """Anything that stops this session from producing depth."""


@contextlib.contextmanager
def _preserve_logging():
    """Undo a third-party ``basicConfig``/``reload(logging)`` at module scope.

    ``FoundationStereo/Utils.py:30`` reconfigures the root logger the moment it
    is imported. The station logs through its own bus, but stdout formatting is
    how every other worker reports, and having one import silently change it
    for the whole process is the kind of action-at-a-distance this repo has
    been bitten by before.
    """
    root = logging.getLogger()
    saved = (list(root.handlers), root.level,
             logging.raiseExceptions, logging.getLogger().propagate)
    try:
        yield
    finally:
        # reload(logging) rebinds the module object's classes, so re-fetch the
        # root logger rather than trusting the reference captured above.
        importlib.import_module("logging")
        cur = logging.getLogger()
        cur.handlers[:] = saved[0]
        cur.setLevel(saved[1])
        logging.raiseExceptions = saved[2]
        cur.propagate = saved[3]


def _sha1_head(path: Path, nbytes: int = 8 << 20) -> str:
    """SHA-1 of the first ``nbytes`` of a file, for provenance.

    The head, not the whole 3 GiB: this runs while the operator waits, and the
    question it answers is "which checkpoint was this run flown with", for
    which a prefix hash of a file nobody edits in place is enough. The length
    is recorded beside it so the two can never be confused for a full digest.
    """
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()


_PATCHED = False


def _import_upstream(src: str):
    """Import FoundationStereo's ``core`` from *src* without keeping ``Utils``.

    Both FoundationStereo and FoundationPose have a top-level ``Utils.py`` and
    import it bare at module import time (``core/extractor.py:17`` here;
    ``estimater.py:10`` and eight more files there). Whoever imports first
    owns ``sys.modules["Utils"]`` and the other gets the wrong module: with
    --fstereo --pose the stereo side loads at startup, and FoundationPose's
    ``from Utils import *`` at the first click would star-import the stereo
    checkout's eight functions and NameError on 31 of its own — object
    tracking dead at the first registration, with nothing in the startup log.

    So the bare name is owned only while this import runs (the two names
    ``core/extractor.py`` binds are bound then), then handed back to whoever
    had it, and this checkout is moved to the END of ``sys.path`` so a later
    bare ``import Utils`` from the FoundationPose root — which sam2_live
    inserts at the FRONT when it loads — finds FoundationPose's. Nothing in
    FoundationStereo's ``core`` imports lazily (grep-verified: every
    ``from core``/``from Utils`` is at module level), so it needs neither the
    name nor the front of the path afterwards. Serialised against the pose
    side by :data:`upstream.UPSTREAM_IMPORT_LOCK`.
    """
    from .upstream import FSTEREO_UTILS_ALIAS, UPSTREAM_IMPORT_LOCK
    with UPSTREAM_IMPORT_LOCK:
        had = sys.modules.pop("Utils", None)
        if src in sys.path:
            sys.path.remove(src)
        sys.path.insert(0, src)
        try:
            with _preserve_logging():
                # Utils.py reconfigures logging at module scope; these two
                # imports are what pull it in.
                from core.foundation_stereo import FoundationStereo
                from core.utils.utils import InputPadder
            mine = sys.modules.get("Utils")
        finally:
            sys.modules.pop("Utils", None)
            if had is not None:
                sys.modules["Utils"] = had
            if src in sys.path:
                sys.path.remove(src)
            sys.path.append(src)
        if mine is not None:
            sys.modules[FSTEREO_UTILS_ALIAS] = mine
    return FoundationStereo, InputPadder


#: sha1[:12] of ``inspect.getsource`` of the upstream bodies the patches
#: replace — FoundationStereo checkout of 2026-09-02 and timm 1.0.29. A
#: mismatch means the transcriptions below may no longer be equivalent.
_PATCH_SOURCE_SHA1 = {
    "normalize_image": "7b6fb3fb1849",
    "pef_forward": "015f017dae07",
    "bilinear_sampler": "5d18c0bd4c7c",
}


def _patch_for_capture(torch, model) -> None:
    """Remove the five host<->device round trips inside the upstream forward.

    Each is a CPU tensor created or read during the forward — which CUDA
    stream capture forbids outright ("Cannot copy between CPU and CUDA tensors
    during CUDA graph capture"), and which in eager mode is a stall the GPU
    sits through on every frame. They are patched on the IMPORTED upstream
    modules, in this process only: the checkout on disk is not edited because
    other tools run it. Every replacement computes the same values; the
    bench checks graph replay against eager on all nine real pairs at ten
    settings and gets np.array_equal every time [측정 2026-09-02,
    rov_gui/tools/fstereo_bench_out/graph.txt].

    1. ``core.foundation_stereo.normalize_image`` builds a torchvision
       ``Normalize`` — and its mean/std tensors — from Python lists per call.
    2. ``model.dx`` is a plain attribute, not a buffer, so ``model.cuda()``
       leaves it on the host and ``core/geometry.py:44`` moves it per call.
    3. ``core.foundation_stereo.autocast`` caches fp16 copies of the weights;
       that cache is illegal inside a capture (``cache_enabled=False``).
    4. timm's ``PositionalEncodingFourier.forward`` builds its mask with
       ``torch.zeros(shape)`` on the CPU and moves it (``edgenext.py:61``).
    5. ``core.utils.utils.bilinear_sampler`` asserts ``torch.unique(ygrid)``
       on EVERY GRU step: a device->host sync eight times per frame. The
       ``H == 1`` half of that assert is kept — it is the half checkable
       without a sync. In eager mode this is the one that stalls every frame.
    """
    global _PATCHED
    # 2. dx lives on the device from the start. Per MODEL, not per module —
    # a second session in this process gets a new model with dx on the host
    # again, and its capture would fail on exactly the copy this removes.
    model.dx = model.dx.cuda()
    if _PATCHED:
        return
    import hashlib
    import inspect
    import torch.nn.functional as F
    from torch.backends import cudnn
    import core.foundation_stereo as fs_mod
    import core.geometry as geo_mod
    import core.utils.utils as uu_mod
    import timm.models.edgenext as en_mod

    # The replacements below are transcriptions of SPECIFIC upstream bodies.
    # Refuse to paste over a body that is not the one they were written
    # against — a newer timm or FoundationStereo would otherwise be silently
    # overridden with old semantics. The caller turns this into "unpatched,
    # eager, said once", not a dead depth source.
    for name, fn, want in (
            ("core.foundation_stereo.normalize_image", fs_mod.normalize_image,
             _PATCH_SOURCE_SHA1["normalize_image"]),
            ("timm.models.edgenext.PositionalEncodingFourier.forward",
             en_mod.PositionalEncodingFourier.forward,
             _PATCH_SOURCE_SHA1["pef_forward"]),
            ("core.utils.utils.bilinear_sampler", uu_mod.bilinear_sampler,
             _PATCH_SOURCE_SHA1["bilinear_sampler"])):
        got = hashlib.sha1(inspect.getsource(fn).encode()).hexdigest()[:12]
        if got != want:
            raise FStereoError(
                f"{name} is not the body these patches were written for "
                f"(source sha1 {got}, expected {want}); upstream or timm "
                f"changed — re-derive the patch before trusting it")

    # 1. normalize_image without a per-call CPU->GPU copy
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def normalize_image(img):
        return ((img / 255.0 - mean) / std).contiguous()

    fs_mod.normalize_image = normalize_image

    # 3. autocast without the weight cache
    class NoCacheAutocast(torch.amp.autocast):
        def __init__(self, enabled=True, dtype=torch.float16,
                     cache_enabled=None):
            super().__init__("cuda", enabled=enabled, dtype=dtype,
                             cache_enabled=False)

    fs_mod.autocast = NoCacheAutocast

    # 4. timm Fourier positional encoding, mask built on the device
    def pef_forward(self, shape):
        device = self.token_projection.weight.device
        dtype = self.token_projection.weight.dtype
        inv_mask = torch.ones(shape, device=device, dtype=torch.bool)  # == ~zeros
        y_embed = inv_mask.cumsum(1, dtype=torch.float32)
        x_embed = inv_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.int64,
                             device=device).to(torch.float32)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return self.token_projection(pos.to(dtype))

    en_mod.PositionalEncodingFourier.forward = pef_forward

    # 5. bilinear_sampler without the syncing assert
    def bilinear_sampler(img, coords, mode="bilinear", mask=False,
                         low_memory=False):
        H, W = img.shape[-2:]
        assert H == 1                        # stereo: the host-checkable half
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1).to(img.dtype)
        with cudnn.flags(enabled=False):
            img = F.grid_sample(img.contiguous(), grid.contiguous(),
                                align_corners=True)
        if mask:
            m = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, m.float()
        return img

    # geometry.py binds the NAME at import (`from core.utils.utils import
    # bilinear_sampler`), so its namespace is what the forward actually calls.
    geo_mod.bilinear_sampler = bilinear_sampler
    uu_mod.bilinear_sampler = bilinear_sampler
    _PATCHED = True


class FStereoSession:
    """Loads FoundationStereo off-thread; turns raw mono pairs into depth.

    Mirrors :class:`rov_gui.perception.session.PoseSession`'s lifecycle on
    purpose — ``start_async`` / ``ready`` / ``error`` / ``close`` — because the
    station already knows how to display exactly that shape of thing.
    """

    def __init__(self, repo=None, ckpt=None, *,
                 iters: int = DEFAULT_ITERS,
                 scale: float = 1.0,
                 z_range_mm: tuple[float, float] = Z_RANGE_MM,
                 graph: bool = DEFAULT_GRAPH,
                 fill: int = DEFAULT_FILL,
                 size: tuple[int, int] | None = None):
        env_repo = os.environ.get("FOUNDATION_STEREO_REPO")
        self.repo = Path(repo or env_repo or DEFAULT_REPO).expanduser()
        self.ckpt = Path(ckpt).expanduser() if ckpt else self.repo / DEFAULT_CKPT
        self.iters = int(iters)
        self.scale = float(scale)
        if not 0.0 < self.scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")
        # An explicit network input size beats the scale. Any aspect ratio is
        # allowed: disparity is a horizontal LENGTH, so only the width ratio
        # has to be undone afterwards (see _from_output).
        self.infer_size = (int(size[0]), int(size[1])) if size else None
        if self.infer_size and min(self.infer_size) < 64:
            raise ValueError(f"size must be at least 64x64, got {self.infer_size}")
        self.z_range_mm = z_range_mm
        self.fill = max(0, int(fill))
        self.graph = bool(graph)      # flips off for the run when capture fails
        self.graph_requested = bool(graph)
        self._last_in_size = None     # (w_in, h_in) the network last saw
        self._graph = None            # torch.cuda.CUDAGraph once captured
        self._graph_key = None        # (padded input shape, iters) it holds
        self._static = None           # (left, right, out) tensors it reads/writes
        self._graph_ms = 0.0
        self._graph_error = ""
        self._on_log = None

        self._lock = threading.Lock()
        self._loading = False
        self._ready = False
        self._closed = False
        self._error = ""
        self._load_s = 0.0
        self._t_load0 = 0.0
        self._loader = None
        self._model = None
        self._padder = None
        self._torch = None
        self._cv2 = None
        self.model_name = ""
        self.ckpt_sha1 = ""
        # Rolling inference cost, for the HUD. Not a frame rate: the worker
        # measures arrivals itself, the same split the pose chip documents
        # (solve ms vs measured Hz answer different questions).
        self._solve_ms = 0.0

    # ------------------------------------------------------------- lifecycle
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        return self._loading

    @property
    def error(self) -> str:
        return self._error

    @property
    def solve_ms(self) -> float:
        return self._solve_ms

    @property
    def load_seconds(self) -> float:
        if self._ready:
            return self._load_s
        return (time.monotonic() - self._t_load0) if self._loading else 0.0

    def start_async(self, on_log=None) -> None:
        """Build the network on a private thread and return immediately.

        Loading is seconds (3 GiB of weights plus a CUDA context). Doing it in
        the worker's tick would stall the thread that has to keep answering
        the enable toggle.
        """
        with self._lock:
            if self._loading or self._ready:
                return
            self._loading = True
            self._on_log = on_log
            # CLEAR THE PREVIOUS FAILURE. `error` is what every consumer polls
            # to decide the feature is dead, so leaving it set makes the retry
            # this method exists for unobservable: the load would run, succeed,
            # log "ready", and the worker would still be short-circuiting on a
            # stale string from the attempt before.
            self._error = ""
            self._t_load0 = time.monotonic()

        def _run():
            try:
                self._load(on_log)
            except Exception as e:                               # noqa: BLE001
                with self._lock:
                    self._error = f"{type(e).__name__}: {e}"
                    self._loading = False
                if on_log and not self._closed:
                    on_log("error", f"fstereo: {self._error}")
            else:
                with self._lock:
                    self._load_s = time.monotonic() - self._t_load0
                    self._loading = False
                    # Closed mid-load must not come back to life.
                    self._ready = not self._closed
                if on_log and self._ready:
                    how = (f"size={self.infer_size[0]}x{self.infer_size[1]}"
                           if self.infer_size else f"scale={self.scale:g}")
                    on_log("info", f"fstereo: {self.model_name} ready in "
                                   f"{self._load_s:.1f} s (iters={self.iters} "
                                   f"{how}, cuda graph "
                                   f"{'on' if self.graph else 'off'})")

        t = threading.Thread(target=_run, name="fstereo-load", daemon=True)
        with self._lock:
            self._loader = t
        t.start()

    def _load(self, on_log=None) -> None:
        """The heavy import + model build. Runs on the loader thread — one
        heavy load at a time in this process (upstream.MODEL_LOAD_LOCK)."""
        from .upstream import MODEL_LOAD_LOCK
        with MODEL_LOAD_LOCK:
            self._load_impl(on_log)

    def _load_impl(self, on_log=None) -> None:
        if not (self.repo / "core" / "foundation_stereo.py").is_file():
            raise FStereoError(
                f"FoundationStereo checkout not found at {self.repo} — run "
                f"'git submodule update --init external/FoundationStereo', "
                f"or pass --fstereo-repo / set FOUNDATION_STEREO_REPO")
        if not self.ckpt.is_file():
            raise FStereoError(
                f"checkpoint not found: {self.ckpt} — the weights are not in "
                f"git (upstream's .gitignore drops pretrained_models/); copy "
                f"the .pth and its cfg.yaml there first (rov_gui/README.md, "
                f"\"FoundationStereo 체크아웃\")")
        cfg_path = self.ckpt.parent / "cfg.yaml"
        if not cfg_path.is_file():
            raise FStereoError(f"no cfg.yaml beside the checkpoint: {cfg_path}")

        # BEFORE torch: dinov2 reads this at module scope (attention.py,
        # block.py, swiglu_ffn.py) and the pinned xformers has no sm_120.
        os.environ.setdefault("XFORMERS_DISABLED", "1")

        src = str(self.repo)

        from ..qt import import_cv2
        self._cv2 = import_cv2()

        if on_log:
            on_log("info", f"fstereo: loading FoundationStereo from {src} "
                           f"(this takes a few seconds)")

        try:
            import torch
            from omegaconf import OmegaConf
            FoundationStereo, InputPadder = _import_upstream(src)
        except ImportError as e:
            raise FStereoError(
                f"{e}. FoundationStereo needs timm, einops and huggingface_hub "
                f"in this interpreter ({sys.executable}):\n"
                f"    {Path(sys.executable).parent / 'pip'} install timm "
                f"einops huggingface_hub") from e

        cfg = OmegaConf.load(cfg_path)
        if "vit_size" not in cfg:
            # 23-51-11 predates the key. run_demo.py injects the same value.
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = self.iters
        model = FoundationStereo(OmegaConf.create(cfg))

        ckpt = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if not torch.cuda.is_available():
            raise FStereoError(
                "no CUDA device — FoundationStereo is not usable on CPU at "
                "any interactive rate")
        model.cuda().eval()
        # Always, not only for the graph: patch 5 is a per-GRU-step sync that
        # stalls the eager path on every frame as well.
        try:
            _patch_for_capture(torch, model)
        except Exception as e:                                   # noqa: BLE001
            # A different upstream/timm layout than the one the patches were
            # written against. Run the code as it is: that costs the graph
            # (capture needs the patches) and the eager syncs, not the run's
            # only depth source.
            self.graph = False
            self._graph_error = f"patch: {type(e).__name__}: {e}"
            if on_log:
                on_log("warn", f"fstereo: could not patch the upstream forward "
                               f"for CUDA graph capture — running it unpatched "
                               f"and eager ({self._graph_error[:120]})")

        with self._lock:
            if self._closed:
                raise FStereoError("session closed during load")
            self._torch = torch
            self._model = model
            self._padder = InputPadder
            self.model_name = f"{cfg['vit_size']} @ {self.ckpt.parent.name}"
            self.ckpt_sha1 = _sha1_head(self.ckpt)

    def close(self) -> None:
        """Drop the model and JOIN the loader.

        Same rule as PoseSession: a daemon thread torn down mid-torch unwinds
        through C++ and aborts the process, so the loader is waited for even
        though its result will be discarded.
        """
        with self._lock:
            self._closed = True
            self._ready = False
            self._model = None
            self._graph = None
            self._static = None
            loader = self._loader
        if loader is not None and loader is not threading.current_thread():
            loader.join(timeout=60.0)
        # Release the weights now rather than at interpreter exit, so a
        # station that turns FS off gets its VRAM back for SAM2/FoundationPose.
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:                                    # noqa: BLE001
                pass

    # -------------------------------------------------------------- compute
    def _say(self, level: str, msg: str) -> None:
        if self._on_log is not None and not self._closed:
            self._on_log(level, msg)

    def _infer_size(self, w: int, h: int) -> tuple[int, int]:
        """The size the network sees for a w x h rectified pair."""
        if self.infer_size is not None:
            return self.infer_size
        if self.scale == 1.0:
            return (w, h)
        return (max(1, int(round(w * self.scale))),
                max(1, int(round(h * self.scale))))

    def _from_output(self, disp_in: np.ndarray, w: int, h: int,
                     w_in: int, h_in: int) -> np.ndarray:
        """Network-size disparity -> disparity at the rectified size.

        Resize back AND rescale by the WIDTH ratio only: disparity is a
        horizontal length, so an anisotropic input (say 224x224 from 640x400)
        changes it by w/w_in and the height ratio not at all.

        INTER_LINEAR here is the reference tool's choice, kept, and it is a
        REAL trade rather than an oversight: interpolating disparity across a
        depth edge fabricates intermediate distances, the same objection the
        NEAREST-only rule makes about depth. NEAREST would keep every value
        honest but visibly staircase a field the network produced as smooth.
        """
        if (w_in, h_in) == (w, h):
            return disp_in
        cv2 = self._cv2
        return cv2.resize(disp_in, (w, h), interpolation=cv2.INTER_LINEAR) * (w / w_in)

    def _eager(self, t_left, t_right):
        torch = self._torch
        # no_grad HERE and not only via the loader's set_grad_enabled: that
        # flag is thread-local and this runs on the worker thread.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             cache_enabled=False):
            return self._model.forward(t_left, t_right, iters=self.iters,
                                       test_mode=True)

    def _capture(self, t_left, t_right, key) -> bool:
        """Record the forward as a CUDA graph for this input shape.

        Warm-up first, on a side stream: the first eager passes do cuDNN
        autotuning, allocator growth and autocast setup, none of which may
        happen inside a capture. ``thread_local`` error mode so that another
        thread's CUDA work (SAM2/FoundationPose under --pose) neither breaks
        this capture nor is broken by it. Failure is final for the session:
        it is logged once, ``graph`` flips off, and every later frame runs
        eager — the same numbers, ~8 ms slower on an idle GPU
        [측정: rov_gui/tools/fstereo_bench_out/graph.txt].
        """
        torch = self._torch
        t0 = time.monotonic()
        prev_stream = None
        try:
            prev_stream = torch.cuda.current_stream()
            sl, sr = t_left.contiguous().clone(), t_right.contiguous().clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._eager(sl, sr)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                out = self._eager(sl, sr)
            torch.cuda.synchronize()
        except Exception as e:                                   # noqa: BLE001
            # torch.cuda.graph.__exit__ has no try/finally: when the body
            # invalidates the capture, capture_end() raises BEFORE the stream
            # context is restored and this thread is left on the capture
            # stream. Put it back and drain, so the eager fallback runs where
            # the rest of the session runs.
            try:
                if prev_stream is not None:
                    torch.cuda.set_stream(prev_stream)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:                                    # noqa: BLE001
                pass
            # torch reports a capture that died inside the body as
            # "operation failed due to a previous error during capture" from
            # capture_end(); the error that actually did it is chained. Name
            # the innermost one, or the log points at the wrong line.
            root = e
            while (root.__cause__ or root.__context__) is not None:
                root = root.__cause__ or root.__context__
            text = f"{type(root).__name__}: {root}"
            if root is not e:
                text = f"{text} (surfaced as {type(e).__name__}: {str(e)[:60]})"
            self._drop_graph(f"capture: {text}",
                             "CUDA graph capture failed, running eager for "
                             "the rest of the run")
            return False
        self._graph, self._graph_key, self._static = g, key, (sl, sr, out)
        self._graph_ms = (time.monotonic() - t0) * 1000.0
        shape = key[0]
        self._say("info", f"fstereo: forward captured as one CUDA graph in "
                          f"{self._graph_ms:.0f} ms (input {shape[-1]}x"
                          f"{shape[-2]}, iters {self.iters})")
        return True

    def _drop_graph(self, error: str, message: str) -> None:
        """Give up on the graph for the rest of the run, once, out loud."""
        self.graph = False
        self._graph = None
        self._graph_key = None
        self._static = None
        self._graph_error = error
        self._say("warn", f"fstereo: {message} — {error[:160]}")

    def _replay(self, t_left, t_right):
        sl, sr, out = self._static
        sl.copy_(t_left)
        sr.copy_(t_right)
        self._graph.replay()
        return out

    def _forward(self, t_left, t_right):
        """One network pass: graph replay when it can be, eager otherwise.

        A replay that raises is treated like a capture that failed: the graph
        is dropped, the frame is recomputed eagerly, and the run stays eager.
        A launch error that only surfaces later, at the ``.cpu()`` in
        :meth:`disparity`, is caught there the same way.
        """
        if self.graph:
            key = (tuple(t_left.shape), self.iters)
            if ((self._graph is not None and self._graph_key == key)
                    or self._capture(t_left, t_right, key)):
                try:
                    return self._replay(t_left, t_right)
                except Exception as e:                           # noqa: BLE001
                    self._drop_graph(f"replay: {type(e).__name__}: {e}",
                                     "CUDA graph replay failed, running eager "
                                     "for the rest of the run")
        return self._eager(t_left, t_right)

    def disparity(self, left_rect: np.ndarray, right_rect: np.ndarray):
        """Rectified gray8 pair -> disparity in pixels at the input resolution.

        Carried over from the reference tool unchanged: 3-channel replication
        of the mono image (the paper supports mono input), 0-255 floats (the
        network normalises internally), pad to a multiple of 32, fp16 autocast.
        """
        torch, cv2 = self._torch, self._cv2
        if self._model is None:
            raise FStereoError("model is not loaded")
        h, w = left_rect.shape[:2]
        w_in, h_in = self._infer_size(w, h)
        if w_in > w or h_in > h:
            # Upstream asserts scale <= 1 for the same reason: the network
            # input can only be shrunk. INTER_AREA on an enlargement is a
            # nearest-neighbour blow-up that the padded tensor, the graph's
            # pool and every kernel then pay for, for no information.
            raise FStereoError(
                f"--fstereo-size {w_in}x{h_in} is larger than the rectified "
                f"pair {w}x{h}; the network input can only be made smaller")
        self._last_in_size = (w_in, h_in)
        if (w_in, h_in) != (w, h):
            left_rect = cv2.resize(left_rect, (w_in, h_in),
                                   interpolation=cv2.INTER_AREA)
            right_rect = cv2.resize(right_rect, (w_in, h_in),
                                    interpolation=cv2.INTER_AREA)

        def to_tensor(img):
            rgb = np.repeat(img[:, :, None], 3, axis=2)
            return torch.as_tensor(rgb).cuda().float()[None].permute(0, 3, 1, 2)

        t_left, t_right = to_tensor(left_rect), to_tensor(right_rect)
        padder = self._padder(t_left.shape, divis_by=32, force_square=False)
        t_left, t_right = padder.pad(t_left, t_right)

        def run():
            d = self._forward(t_left, t_right)
            # .cpu() copies out synchronously, so a graph's static output
            # buffer is safe to read here even though the next replay
            # overwrites it. It is also where an asynchronous launch error
            # from a replay surfaces.
            d = padder.unpad(d.float())
            return d.cpu().numpy().reshape(d.shape[-2], d.shape[-1])

        try:
            disp = run()
        except Exception as e:                                   # noqa: BLE001
            if self._graph is None:
                raise
            self._drop_graph(f"replay: {type(e).__name__}: {e}",
                             "CUDA graph replay failed, running eager for "
                             "the rest of the run")
            disp = run()
        return self._from_output(disp, w, h, w_in, h_in)

    def infer(self, left_raw: np.ndarray, right_raw: np.ndarray, rig,
              out_size=None) -> dict:
        """Raw mono pair + rig -> uint16 mm depth on the colour grid.

        Returns a dict with the depth map and the two valid-pixel percentages
        that the panel HUD needs to keep them apart: ``valid_native`` is what
        the network matched, ``valid_out`` is what the projection onto the
        colour grid actually landed, and ``filled_out`` is what
        :func:`fill_scatter_gaps` then closed. The second number is
        structurally lower than the first and saying so is the only thing that
        stops it being read as the network failing; the third is kept separate
        from the second for the opposite reason — so a repaired pixel is never
        counted as a measured one.
        """
        from c3_camera.host_depth import (depth_from_disparity,
                                          fill_scatter_gaps,
                                          resize_depth_nearest,
                                          warp_depth_to_color)
        cv2 = self._cv2
        t0 = time.monotonic()

        if not rig.rectifies:
            raise FStereoError("this rig has no rectification maps")
        ml, mr = rig.map_left, rig.map_right
        # INTER_LINEAR: this resamples INTENSITY, which is what rectification
        # is for. The NEAREST-only rule applies to depth, below.
        l_rect = cv2.remap(left_raw, ml[0], ml[1], cv2.INTER_LINEAR)
        r_rect = cv2.remap(right_raw, mr[0], mr[1], cv2.INTER_LINEAR)

        disp = self.disparity(l_rect, r_rect)
        depth_native = depth_from_disparity(disp, rig.k_mm_px, self.z_range_mm)

        warped = warp_depth_to_color(depth_native, rig)
        # MEASURED coverage, before anything is filled in. The two numbers are
        # kept apart all the way to the chip: a filled pixel is a neighbour's
        # millimetres shown where the projection landed nothing, and folding it
        # into "valid" would turn a resampling repair into a measurement.
        measured = float(np.count_nonzero(warped)) / float(warped.size)
        depth_out = fill_scatter_gaps(warped, self.fill)
        filled = float(np.count_nonzero(depth_out)) / float(depth_out.size) - measured
        if out_size is not None:
            # NEAREST, so the two fractions survive the resize: measured 95.31%
            # at 400x250 vs 95.32% at 640x360 on a real frame.
            depth_out = resize_depth_nearest(depth_out, out_size)

        dt = time.monotonic() - t0
        self._solve_ms = dt * 1000.0
        return {
            "depth_mm": depth_out,
            # The map BEFORE the colour-grid warp and the gap fill: uint16 mm on
            # the host rectified-left 640x400 grid (rig.R1 / rig.P1). The policy
            # observation builder (perception/policy_obs.py, grid rect_left)
            # re-warps THIS onto the training target grid — the filled,
            # colour-aligned `depth_mm` would carry a repair and a second
            # resampling the training data never saw.
            "depth_native": depth_native,
            "rect_left": l_rect,
            "valid_native": 100.0 * float(np.count_nonzero(depth_native))
            / float(depth_native.size),
            "valid_out": 100.0 * measured,
            "filled_out": 100.0 * filled,
            "solve_ms": self._solve_ms,
        }

    # ------------------------------------------------------------- metadata
    def describe(self) -> dict:
        """Provenance for the run folder. Written whether or not it ever ran."""
        return {
            "depth_source": "foundation_stereo",
            "repo": str(self.repo),
            "ckpt": str(self.ckpt),
            "ckpt_sha1_first_8mib": self.ckpt_sha1 or None,
            "model": self.model_name or None,
            "iters": self.iters,
            "scale": self.scale,
            # An explicit size overrides the scale, so say which one governed.
            "scale_applied": self.infer_size is None,
            "infer_size": list(self.infer_size) if self.infer_size else None,
            # What the network ACTUALLY saw, once a frame has: None before.
            "infer_size_actual": (list(self._last_in_size)
                                  if self._last_in_size else None),
            # Three fields because each answers a different question: was the
            # graph asked for (the CLI), is one captured right now (the first
            # frame does it, so a record written before any pair arrived says
            # false here and that is not a failure), and did it fail.
            "cuda_graph_requested": self.graph_requested,
            "cuda_graph": self._graph is not None,
            "cuda_graph_input": (list(self._graph_key[0]) if self._graph_key
                                 else None),
            "cuda_graph_error": self._graph_error or None,
            "scatter_fill_iters": self.fill,
            "z_range_mm": list(self.z_range_mm),
            # The device correction is NOT applied to this stream: 0.64 was
            # fitted to the on-device block matcher's disparity bias, and this
            # path computes its own disparity with a rectification-derived
            # fx_rect. Multiplying by it would correct a corrected number.
            "depth_scale_applied": 1.0,
            "rectified": "host (cv2.stereoRectify via c3_camera.host_depth)",
        }
