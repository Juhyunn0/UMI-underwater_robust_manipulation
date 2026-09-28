"""dp_policy.py — the trained diffusion policy as a station-side inference session.

What this is
------------
``DpPolicySession`` loads ONE checkpoint of UMI's
``DiffusionTransformerTimmPolicy`` (TransformerObsEncoder on a timm resnet34 +
TransformerForActionDiffusion + DDIM) and answers ``predict(obs) -> (action
(16, D) float32, infer_ms)`` in the checkpoint's action contract: 16 knots
relative to the TCP pose at the newest observation, one knot every ``obs_dt``
(= down_sample_steps / dataset fps), where D and the column meaning are the
checkpoint's ``action_repr`` (``rov_gui.state.ACTION_REPR_*``):
``pos_yaw_width`` D = 5 [dx, dy, dz, dyaw, gripper_width_m] (the 2026-09-07
retrain, what the station flies), ``pos_rpy_width`` D = 7 [dx, dy, dz, dyaw,
droll, dpitch, gripper_width_m] (the 6-DoF variant, 2026-09-26: columns 0:4
and 6 bit-identical to the 5-dim ones, roll/pitch labelled through the same
C3-mount rotation and declared in ``shape_meta.action.rpy_convention`` --
``RPY_CONVENTION_STATION`` below; whether the station TRACKS them is config
``policy.attitude_track``, never this file's concern) or the legacy
``pose10d`` D = 10 [pos(3), rot6d(6), gripper_width_m] (kept loadable so old
records can be recomposed; the controller refuses to ARM on it,
control/workers.py ``_policy_refusal``).
Everything about frames, anchoring, timing and safety lives OUTSIDE this file
(``control/policy_frames.py``, ``backends/policy.py``, ``MpcWorker``); this
file's only job is to be the trained network, bit-for-bit, in the station's
interpreter.

Why it is a re-implementation and not ``hydra.utils.instantiate(cfg.policy)``
---------------------------------------------------------------------------
The station runs in ``rovgui-pose`` (torch 2.11, timm 1.0.29, diffusers 0.40)
while the checkpoint was trained in ``umi2`` (torch 2.9, timm 0.9.7,
diffusers 0.18). Three upstream modules cannot be imported here:

* ``diffusion_policy.model.common.normalizer`` references ``zarr.Array`` at
  import time, and ``rovgui-pose`` has no real zarr — the repo's own ``zarr/``
  data directory shadows it as a namespace package. So the LinearNormalizer
  is re-implemented (:class:`_Normalizer`: ``x * scale + offset`` forward,
  ``(x - offset) / scale`` back, per key, on the flattened last dimension —
  the exact arithmetic of ``SingleFieldLinearNormalizer._normalize``).
* ``diffusion_policy.policy.base_image_policy`` pulls the normalizer in.
* ``DiffusionTransformerTimmPolicy`` itself pulls both in. Its two inference
  methods (``predict_action`` and ``conditional_sample``) are transcribed
  into :meth:`DpPolicySession.predict` line for line; the parity tool
  ``rov_gui/tools/dp_policy_parity.py`` checks the transcription against
  outputs computed by the upstream class in the training environment
  (``rov_gui/tests/data/dp_policy_ref.npz``).

Only two upstream modules are imported — the encoder and the denoiser — under
``perception/upstream.UPSTREAM_IMPORT_LOCK`` with the UMI checkout appended at
the END of ``sys.path`` (spec v2 A20): the checkout also carries top-level
names (``umi``, ``scripts``...) that must never shadow the station's own.

Architecture facts the loader relies on (checkpoint cfg, verified 2026-09-02)
--------------------------------------------------------------------------
* ``obs_encoder``: ``resnet34.a1_in1k``, ``pretrained: true`` and
  ``use_group_norm: true`` — upstream only swaps BatchNorm for GroupNorm when
  ``use_group_norm and not pretrained``, so the TRAINED net kept BatchNorm.
  The encoder is therefore built with ``pretrained=False`` (no timm download;
  the weights come from the checkpoint) and ``use_group_norm`` derived the
  same way upstream does, which keeps the state-dict keys identical and lets
  ``load_state_dict(strict=True)`` be the proof of architectural identity.
* transforms: ``RandomCrop(ratio 0.95) -> Resize(224, antialias) ->
  RandomRotation(+-5 deg)``, NO Normalize. ``eval_transforms`` picks what runs
  at inference: ``"center"`` (default, spec v2 A19) = deterministic
  ``CenterCrop(int(224*0.95)) -> Resize(224, antialias=True)`` — inside the
  training distribution, no rotation; ``"train"`` = upstream's random crop
  and rotation exactly as the cfg lists them (what UMI's ``eval_real`` does);
  ``"identity"`` = no transform (parity/tests only, never for flight).
* ``noise_scheduler``: DDIM, 50 train steps, squaredcos_cap_v2, clip_sample,
  set_alpha_to_one, steps_offset 0, epsilon; ``num_inference_steps`` 16.
* ``state_dicts.ema_model`` is what ``selected.json`` says to deploy. It
  splits into ``obs_encoder.*``, ``model.*`` and ``normalizer.*``; the first
  two are strict-loaded, the third feeds :class:`_Normalizer`, and any other
  key is an error (a different policy class than this loader understands).

Timing and threads
------------------
The forward runs on the session's OWN ``torch.cuda.Stream`` so it never
serialises behind FoundationStereo's graph replay on the default stream.
``torch.no_grad`` is applied per call (grad mode is thread-local; the worker
thread that calls ``predict`` is not the thread that loaded the model). Load
ends with one warm-up predict so the first live plan does not pay for cuDNN
autotuning and lazy kernel loading (spec v2 A7). ``predict`` is serialised by
a lock — one forward at a time per session.

``torch`` is imported inside :meth:`DpPolicySession.load` only. Importing this
module must stay free on a machine without torch, like the rest of
``rov_gui.perception``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import operator
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..state import (ACTION_DIM_BY_REPR, ACTION_REPR_BY_DIM,
                     ACTION_REPR_POS_RPY_WIDTH, ACTION_REPR_POS_YAW_WIDTH,
                     ACTION_REPR_POSE10D, POLICY_CKPT_DEFAULT, POLICY_UMI_REPO)

#: The UMI checkout whose ``diffusion_policy`` package built the checkpoint.
#: ONE source: ``rov_gui.state.POLICY_UMI_REPO`` (the geometry policy block,
#: the backend fallback block and the tools all read the same constant).
DEFAULT_UMI_REPO = POLICY_UMI_REPO

#: The deployed checkpoint (``selected.ckpt``; see ``selected.json`` beside it
#: for the epoch and the evidence). ONE source: ``rov_gui.state.
#: POLICY_CKPT_DEFAULT`` -- a retrain is selected by editing that line and the
#: two YAML ``policy.ckpt`` lines, nothing here.
DEFAULT_CKPT = POLICY_CKPT_DEFAULT

#: Inference-time image transform choices (see module docstring).
EVAL_TRANSFORMS = ("center", "train", "identity")

#: Which state dict inside the checkpoint to fly.
#:
#: ``ema_model`` is what ``selected.json`` names and what shipped, but on the
#: 2026-09-01 checkpoint it is measurably WORSE on held-out actions, and the
#: reason is a defect rather than a property of EMA:
#:
#: * ``EMAModel.step`` iterates ``named_parameters()`` only, so BUFFERS are
#:   never averaged — BatchNorm's ``running_mean`` / ``running_var`` are
#:   buffers, not parameters;
#: * the encoder still HAS BatchNorm, because
#:   ``transformer_obs_encoder.py:120`` reads
#:   ``if use_group_norm and not pretrained:`` and the config sets both to
#:   true, so the GroupNorm substitution was silently skipped.
#:
#: So the EMA copy carries the timm ImageNet statistics unchanged
#: [측정: num_batches_tracked — pretrained 374,981, ckpt ema_model 374,983,
#: ckpt model 395,128], and at eval BatchNorm normalises depth with them.
#:
#: Held-out action MSE over the full validation split (episodes 6/32/48/56 by
#: ``get_val_mask(75, 0.05, 42)``, 219 windows, DDIM 8 = the setting flown on
#: 2026-09-03) [측정 2026-09-06]::
#:
#:     ema_model                     0.001535   pos RMSE 30.7 mm
#:     model                         0.000842   pos RMSE 20.5 mm
#:     ema params + trained buffers  0.000896   pos RMSE 20.9 mm
#:
#: The third row is the decomposition: swapping ONLY the 108 BatchNorm buffers
#: recovers 92 % of the gap, so the AVERAGING is fine and the statistics were
#: the whole problem.
#:
#: ``ema_model`` stays the default so an unflagged run still reproduces every
#: record written before 2026-09-06. The choice is in ``describe()`` and in
#: the run meta, and runs on different weights must not be pooled.
WEIGHT_SOURCES = ("ema_model", "model")


def _select_weights(state_dicts: dict, which: str, log=None) -> dict:
    """The requested state dict, or a refusal naming what the file holds."""
    if which not in state_dicts:
        raise ValueError(
            f"the checkpoint has no state_dicts[{which!r}]; it holds "
            f"{sorted(state_dicts)}. Pick one with --policy-weights.")
    # Which copy flew is a RECORD BOUNDARY either way, so say it out loud. Since
    # 2026-09-07 the DEPLOYED choice is "model": the EMA copy carries the BatchNorm
    # defect (KNOWN_ISSUES.md) and measures far worse on the same epoch [측정:
    # 219-window held-out, 5-dim ep195 — ema_model 33.0 mm vs model 17.3 mm].
    if log is not None:
        note = ("" if which == "model" else
                " (NOT the deployed 'model' copy — the EMA copy carries the"
                " BatchNorm defect)")
        log(f"DP policy: flying state_dicts.{which}{note} — records on "
            f"different weights must not be pooled")
    return state_dicts[which]

#: The normalizer / state-dict prefixes of ``state_dicts.ema_model``.
_PFX_ENC = "obs_encoder."
_PFX_MODEL = "model."
_PFX_NORM = "normalizer.params_dict."

#: sha1 of the first 8 MiB of the checkpoint, like fstereo's — enough to tell
#: two checkpoints apart, cheap enough to do at every start.
SHA1_HEAD_BYTES = 8 << 20

#: Stub chunk: straight ahead along TCP +z (forward in the camera optical
#: frame) at this speed, jaw held at this width. The width is the dataset's
#: OPEN level [측정: /home/bdml/Desktop/data collection/slam/actions_summary.json
#: open_level_m 0.062-0.069] so a stub plan with gripper enabled reads as
#: "open" through the same thresholds a real plan would.
STUB_SPEED_M_S = 0.05
STUB_WIDTH_M = 0.069

#: The rot6d of the identity rotation (first two ROWS of I, umi.common.pose_util).
ROT6D_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

#: The stub contract's ``action_range`` per representation -- a PLAUSIBLE
#: fixed range (pos +-0.5 m, yaw +-pi, rot6d +-1, width [0.02, 0.09] m), not
#: a measurement: the real session reads the normalizer's input_stats and a
#: consumer checking "every column inside action_range" must see the same
#: keys on the stub. Fixed so the stub's own outputs (<= 0.05 m, 0.069 m)
#: always sit inside it.
STUB_ACTION_RANGE = {
    ACTION_REPR_POS_YAW_WIDTH: ([-0.5, -0.5, -0.5, -math.pi, 0.02],
                                [0.5, 0.5, 0.5, math.pi, 0.09]),
    # 6-DoF variant (2026-09-26): roll/pitch columns +-pi/2 -- the encoder's
    # ZYX pitch is asin-bounded and the dataset build asserts |dpitch| < 60
    # deg [예측], so +-pi/2 is a plausible stub bound, not a measurement.
    ACTION_REPR_POS_RPY_WIDTH: ([-0.5, -0.5, -0.5, -math.pi, -math.pi / 2,
                                 -math.pi / 2, 0.02],
                                [0.5, 0.5, 0.5, math.pi, math.pi / 2,
                                 math.pi / 2, 0.09]),
    ACTION_REPR_POSE10D: ([-0.5] * 3 + [-1.0] * 6 + [0.02],
                          [0.5] * 3 + [1.0] * 6 + [0.09]),
}

#: The roll/pitch label CONVENTION a ``pos_rpy_width`` checkpoint must declare
#: in ``shape_meta.action.rpy_convention`` (task yaml umi_depth_7d.yaml /
#: umi_rgbd_7d.yaml; design D2 2026-09-26): (droll, dpitch, dyaw) are the ZYX
#: Euler angles of ``R_bt @ R_rel @ R_bt^T`` -- the relative TCP rotation
#: carried into the body(FRD) frame of a vehicle wearing the camera on the C3
#: mount -- in the column order [dx, dy, dz, dyaw, droll, dpitch, width]. The
#: station decodes exactly this (control/policy_frames.decode_pos_rpy), so a
#: checkpoint declaring anything else must not be flown: backends/policy.py
#: ``_check_mount`` compares the contract's dict with this one (order-
#: sensitive on ``columns``). Kept here (stdlib+numpy module) so the backend,
#: the tools and the tests read ONE constant; control/policy_frames may
#: re-export it.
RPY_CONVENTION_STATION = {
    "order": "zyx",
    "frame": "body_frd_via_R_bt",
    "columns": ["dx", "dy", "dz", "dyaw", "droll", "dpitch", "width"],
}

#: The stub session spellings ``make_policy_session`` understands (the
#: ``ckpt`` string, case-insensitive), each with the StubPolicySession kwargs
#: it implies. ``stub`` = the flown 5-dim contract (unchanged); ``stub7`` = a
#: (16, 7) ``pos_rpy_width`` chunk with roll/pitch columns identically 0
#: (the dropped-and-logged wiring test); ``stub_rp`` = the same with a
#: 5 deg [예측] pitch ramp reaching the last knot, so demo_e2e can push a
#: KNOWN non-zero attitude request through compose -> filter -> stitcher ->
#: NedPlan -> xref -> allocation (design S8). ``is_stub_ckpt`` is the ONE
#: predicate the backend uses in place of ``== "stub"``.
STUB_CKPT_VARIANTS = {
    "stub": {},
    "stub7": {"action_repr": ACTION_REPR_POS_RPY_WIDTH},
    "stub_rp": {"action_repr": ACTION_REPR_POS_RPY_WIDTH, "pitch_ramp_deg": 5.0},
}


def is_stub_ckpt(ckpt) -> bool:
    """True when ``ckpt`` names a stub session (``STUB_CKPT_VARIANTS``),
    not a checkpoint file. ``None`` / "" are not stubs."""
    return str(ckpt or "").strip().lower() in STUB_CKPT_VARIANTS


def rpy_convention_of(action_meta: dict) -> Optional[dict]:
    """``shape_meta.action.rpy_convention`` as a plain dict (``order`` /
    ``frame`` strings, ``columns`` a list of str), None when the checkpoint
    does not declare one (every 5-dim / pose10d checkpoint), ``ValueError``
    when it is present but not a mapping -- the backend decides whether an
    absent convention is acceptable for the checkpoint's action_repr."""
    v = action_meta.get("rpy_convention")
    if v is None:
        return None
    try:
        items = dict(v).items()
    except (TypeError, ValueError):
        raise ValueError("shape_meta.action.rpy_convention must be a mapping "
                         f"{{order, frame, columns}}, got {v!r}") from None
    out: dict = {}
    for k, x in items:
        if isinstance(x, (list, tuple)):
            out[str(k)] = [str(c) for c in x]
        else:
            out[str(k)] = str(x)
    return out


_INFER_HISTORY = 1000


def resolve_action_repr(action_meta: dict, action_dim: int) -> str:
    """The checkpoint's action representation (``ACTION_REPR_*``).

    ``shape_meta.action.action_repr`` when the checkpoint declares it (the
    2026-09-07 task yaml does), else inferred from ``action_dim`` through
    ``ACTION_REPR_BY_DIM``. ``ValueError`` when neither is known, when the
    declared name is not in the vocabulary, or when the two DISAGREE -- a
    checkpoint whose declared representation does not match its own width is
    mislabelled and must not be decoded by either rule.
    """
    declared = action_meta.get("action_repr")
    inferred = ACTION_REPR_BY_DIM.get(int(action_dim))
    if declared is None:
        if inferred is None:
            raise ValueError(
                f"action dim {int(action_dim)} is not a known representation "
                f"({sorted(ACTION_REPR_BY_DIM)}) and shape_meta.action has no "
                f"action_repr")
        return inferred
    declared = str(declared)
    if declared not in ACTION_DIM_BY_REPR:
        raise ValueError(f"shape_meta.action.action_repr {declared!r} is not "
                         f"one of {sorted(ACTION_DIM_BY_REPR)}")
    if int(action_dim) != ACTION_DIM_BY_REPR[declared]:
        raise ValueError(
            f"shape_meta.action.action_repr {declared!r} implies dim "
            f"{ACTION_DIM_BY_REPR[declared]} but shape_meta.action.shape is "
            f"[{int(action_dim)}]" + (f" ({inferred})" if inferred else ""))
    return declared


def identity_normaliser_columns(action_repr: str, scale, offset) -> list:
    """The action-normaliser columns a checkpoint of ``action_repr`` must
    be REFUSED for (empty = fine).

    THE TRAINING-SIDE TRAP: upstream ``UmiDataset.get_normalizer``
    range-normalises the first 3 and the last column and leaves the middle
    as IDENTITY (scale 1, offset 0) -- right for rot6d, silently wrong for
    an angle column (dyaw of ``pos_yaw_width``, dyaw / droll / dpitch of
    ``pos_rpy_width``, 2026-09-26). With DDIM ``clip_sample`` an identity
    column is clipped to +-1 rad in normalised space and the angle was
    never learned on the [-1, 1] range the other columns use. So for every
    representation OTHER than ``pose10d`` an identity column means the
    checkpoint was trained through the old slice and must not fly; the
    legacy ``pose10d`` keeps its six identity rot6d columns by design.
    """
    if action_repr == ACTION_REPR_POSE10D:
        return []
    scale = np.asarray(scale, float).reshape(-1)
    offset = np.asarray(offset, float).reshape(-1)
    return np.flatnonzero((scale == 1.0) & (offset == 0.0)).tolist()


def _yaw_axis_R(value) -> Optional[list]:
    """``shape_meta.action.yaw_axis_R_frd_cam`` (9 floats row-major, or a
    3x3) -> a 3x3 nested list; None when absent; ``ValueError`` when
    malformed (the backend compares it with hw_nav's R_frd_cam)."""
    if value is None:
        return None
    R = np.asarray(value, dtype=float)
    if R.size != 9 or not np.all(np.isfinite(R)):
        raise ValueError("shape_meta.action.yaw_axis_R_frd_cam must be 9 "
                         f"finite floats, got {value!r}")
    return R.reshape(3, 3).tolist()


def _pickled_int(value, loads) -> Optional[int]:
    """``payload['pickles'][k]`` -> int. The values are dill-pickled bytes
    (upstream ``BaseWorkspace.save_checkpoint``), unpickled with the SAME
    module ``torch.load(pickle_module=dill, weights_only=False)`` already
    trusted for the whole payload; a bare int is taken as is; anything else
    (or a failed unpickle) is None -- these two are provenance, not flight."""
    try:
        if isinstance(value, (bytes, bytearray)):
            value = loads(bytes(value))
        if isinstance(value, (bool, np.bool_)):
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if hasattr(value, "item"):
            v = value.item()
            return int(v) if isinstance(v, int) else None
    except Exception:                                        # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------
# config helpers
# --------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def safe_eval(expr) -> float:
    """Arithmetic-only stand-in for the ``${eval:...}`` OmegaConf resolver.

    UMI's ``train.py`` registers Python's ``eval``; the checkpoint's config
    only ever uses it for arithmetic on numbers (``${eval:'${...} * 2'}``),
    and a checkpoint is data, not code — there is no reason to hand it the
    interpreter. Same function as ``rov_gui/tools/dp_policy_reference.py``
    (kept in both so the perception package does not import a tool).
    """
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression node: {ast.dump(n)}")
    return ev(ast.parse(str(expr), mode="eval"))


def sha1_head(path, nbytes: int = SHA1_HEAD_BYTES) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()


def cfg_fingerprint(shape_meta: dict, policy_cfg: dict) -> str:
    """sha1 of the RESOLVED shape_meta + policy cfg as canonical JSON — the
    architecture/contract identity, independent of the weights."""
    blob = json.dumps({"shape_meta": shape_meta, "policy": policy_cfg},
                      sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def read_zip_zattrs(zip_path) -> Optional[dict]:
    """The root ``.zattrs`` of a zarr ZipStore, via ``zipfile`` only (no zarr
    import — ``rovgui-pose`` has none). None when the zip does not exist."""
    p = Path(zip_path)
    if not p.is_file():
        return None
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read(".zattrs").decode("utf-8"))


# --------------------------------------------------------------------------
# normalizer (re-implementation of upstream LinearNormalizer, inference part)
# --------------------------------------------------------------------------
class _Normalizer:
    """Per-key affine normalizer, backend-agnostic (numpy or torch arrays).

    ``normalize(key, x) = x * scale + offset`` and ``unnormalize(key, x) =
    (x - offset) / scale`` on ``x.reshape(-1, D)`` where ``D = scale.shape[0]``,
    reshaped back — the arithmetic of upstream
    ``SingleFieldLinearNormalizer._normalize`` (module docstring says why it
    is not imported). ``scale``/``offset`` and ``x`` must share a backend; the
    session keeps a torch copy on the device (:meth:`to_torch`) and the
    numpy original for :meth:`describe`.

    The image key ships with scale 1 / offset 0 (upstream
    ``get_image_identity_normalizer``); it goes through the same formula and
    IS the identity — nothing special-cased, so a checkpoint whose image
    normalizer is not identity would still be honoured.
    """

    def __init__(self, params: Dict[str, dict]):
        for key, p in params.items():
            if "scale" not in p or "offset" not in p:
                raise ValueError(f"normalizer key {key!r}: needs scale+offset")
            if tuple(p["scale"].shape) != tuple(p["offset"].shape) \
                    or len(p["scale"].shape) != 1:
                raise ValueError(
                    f"normalizer key {key!r}: scale {tuple(p['scale'].shape)} "
                    f"and offset {tuple(p['offset'].shape)} must be equal 1-D")
        self._p = dict(params)

    @classmethod
    def from_state_dict(cls, sd: dict, prefix: str = _PFX_NORM) -> "_Normalizer":
        """Group ``<prefix><key>.{scale,offset,input_stats.*}`` entries."""
        params: Dict[str, dict] = {}
        for full, v in sd.items():
            if not full.startswith(prefix):
                continue
            rest = full[len(prefix):]
            key, field = rest.split(".", 1)
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            d = params.setdefault(key, {})
            if field.startswith("input_stats."):
                d.setdefault("input_stats", {})[field[len("input_stats."):]] = arr
            else:
                d[field] = arr
        if not params:
            raise ValueError(f"no {prefix}* entries in the state dict")
        return cls(params)

    @property
    def keys(self):
        return list(self._p.keys())

    def dim(self, key: str) -> int:
        return int(self._p[key]["scale"].shape[0])

    def input_stats(self, key: str) -> dict:
        return dict(self._p[key].get("input_stats", {}))

    def params(self, key: str) -> Tuple[object, object]:
        return self._p[key]["scale"], self._p[key]["offset"]

    def normalize(self, key: str, x):
        scale, offset = self.params(key)
        shape = x.shape
        y = x.reshape(-1, scale.shape[0]) * scale + offset
        return y.reshape(shape)

    def unnormalize(self, key: str, x):
        scale, offset = self.params(key)
        shape = x.shape
        y = (x.reshape(-1, scale.shape[0]) - offset) / scale
        return y.reshape(shape)

    def normalize_dict(self, obs: dict) -> dict:
        return {k: self.normalize(k, v) for k, v in obs.items()}

    def to_torch(self, torch, device, dtype) -> "_Normalizer":
        out = {}
        for k, p in self._p.items():
            q = {"scale": torch.as_tensor(np.asarray(p["scale"]), dtype=dtype,
                                          device=device),
                 "offset": torch.as_tensor(np.asarray(p["offset"]), dtype=dtype,
                                           device=device)}
            if "input_stats" in p:
                q["input_stats"] = p["input_stats"]
            out[k] = q
        return _Normalizer(out)


# --------------------------------------------------------------------------
# upstream import (A20)
# --------------------------------------------------------------------------
_ALLOWED_UPSTREAM = (
    "diffusion_policy.model.vision.transformer_obs_encoder",
    "diffusion_policy.model.diffusion.transformer_for_action_diffusion",
)
_FORBIDDEN_UPSTREAM = (
    "diffusion_policy.model.common.normalizer",
    "diffusion_policy.policy.base_image_policy",
)


def _import_upstream(repo: str):
    """Import the encoder and denoiser classes from the UMI checkout at
    ``repo`` — and nothing else from it.

    The checkout goes at the END of ``sys.path`` (never the front: it carries
    top-level ``umi``, ``scripts``, ``ray_utils``... that must not shadow the
    station's names, and the FoundationStereo/FoundationPose ``Utils``
    collision in ``fstereo._import_upstream`` shows what a front insert
    costs). ``diffusion_policy`` is a namespace package there (no
    ``__init__``), so importing the two modules pulls in only their own
    imports (torch, timm, torchvision, the mixin, pytorch_util, positional
    embedding). Serialised against the other upstream imports by
    :data:`upstream.UPSTREAM_IMPORT_LOCK`; asserts afterwards that the
    forbidden modules (zarr shadowing) did not sneak in and that the bare
    ``Utils`` name is exactly what it was.
    """
    from .upstream import UPSTREAM_IMPORT_LOCK
    repo = str(Path(repo))
    with UPSTREAM_IMPORT_LOCK:
        had_utils = sys.modules.get("Utils")
        if repo in sys.path:
            sys.path.remove(repo)
        sys.path.append(repo)
        try:
            enc_mod = importlib.import_module(_ALLOWED_UPSTREAM[0])
            den_mod = importlib.import_module(_ALLOWED_UPSTREAM[1])
        finally:
            if sys.modules.get("Utils") is not had_utils:
                sys.modules.pop("Utils", None)
                if had_utils is not None:
                    sys.modules["Utils"] = had_utils
        for name in _FORBIDDEN_UPSTREAM:
            if name in sys.modules:
                raise ImportError(
                    f"{name} was imported — it references zarr at import "
                    f"time and must stay out of this process (dp_policy.py "
                    f"docstring); check the upstream checkout at {repo}")
        for mod in (enc_mod, den_mod):
            f = getattr(mod, "__file__", "") or ""
            if not f.startswith(repo):
                raise ImportError(
                    f"{mod.__name__} resolved to {f!r}, not inside {repo} — "
                    f"another diffusion_policy is earlier on sys.path")
    return enc_mod.TransformerObsEncoder, den_mod.TransformerForActionDiffusion


# --------------------------------------------------------------------------
# the real session
# --------------------------------------------------------------------------
def _obs_shapes_from_meta(shape_meta: dict):
    """(image_key, image_shape, lowdim_shapes {key: (D,)}, obs_keys, horizon)."""
    image_key, image_shape, lowdim, keys, horizons = None, None, {}, [], set()
    for key, attr in shape_meta["obs"].items():
        keys.append(key)
        horizons.add(int(attr["horizon"]))
        shape = tuple(int(s) for s in attr["shape"])
        if attr.get("type", "low_dim") == "rgb":
            if image_key is not None:
                raise ValueError("this loader expects exactly one image key")
            image_key, image_shape = key, shape
        else:
            lowdim[key] = shape
    if image_key is None:
        raise ValueError("shape_meta has no rgb key")
    if len(horizons) != 1:
        raise ValueError(f"mixed obs horizons {sorted(horizons)} unsupported")
    return image_key, image_shape, lowdim, keys, horizons.pop()


class DpPolicySession:
    """One trained policy, loaded once, queried many times (module docstring).

    Lifecycle: ``start_async(on_log)`` (or synchronous :meth:`load`) ->
    ``ready`` -> ``predict`` ... -> ``close``. ``error`` carries the load
    failure text; ``loading`` is True while the loader thread runs.
    """

    def __init__(self, ckpt=DEFAULT_CKPT, repo=DEFAULT_UMI_REPO, *,
                 num_inference_steps: int = 16,
                 eval_transforms: str = "center",
                 device: str = "cuda",
                 dataset_fps: float = 30.0,
                 weights: str = "ema_model"):
        if weights not in WEIGHT_SOURCES:
            raise ValueError(f"weights must be one of {WEIGHT_SOURCES}, "
                             f"got {weights!r}")
        if eval_transforms not in EVAL_TRANSFORMS:
            raise ValueError(f"eval_transforms must be one of "
                             f"{EVAL_TRANSFORMS}, got {eval_transforms!r}")
        if int(num_inference_steps) < 1:
            raise ValueError("num_inference_steps must be >= 1")
        self.ckpt = str(ckpt)
        self.repo = str(repo)
        self.num_inference_steps = int(num_inference_steps)
        self.eval_transforms = str(eval_transforms)
        self.weights = str(weights)
        self.device_str = str(device)
        self.dataset_fps = float(dataset_fps)

        self.ready = False
        self.loading = False
        self.error = ""
        self.load_seconds = 0.0
        self.warmup_ms = float("nan")

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._torch = None
        self._device = None
        self._stream = None
        self._encoder = None
        self._model = None
        self._scheduler = None
        self._norm_np: Optional[_Normalizer] = None
        self._norm_t: Optional[_Normalizer] = None
        self._contract: dict = {}
        self._desc: dict = {}
        self._infer_ms = deque(maxlen=_INFER_HISTORY)
        self._n_predict = 0

    # -- lifecycle ---------------------------------------------------------
    def start_async(self, on_log: Optional[Callable[[str], None]] = None) -> None:
        """Load on a daemon thread; poll ``ready`` / ``error``."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.loading = True
        self._thread = threading.Thread(
            target=self._load_guarded, args=(on_log,),
            name="dp-policy-load", daemon=True)
        self._thread.start()

    def _load_guarded(self, on_log) -> None:
        try:
            self.load(on_log)
        except Exception as e:                                   # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.ready = False
            if on_log:
                on_log(f"DP policy load FAILED: {self.error}")
        finally:
            self.loading = False

    def load(self, on_log: Optional[Callable[[str], None]] = None) -> None:
        """Synchronous load (raises on failure). Ends with a warm-up predict.
        One heavy model load at a time in this process
        (perception/upstream.MODEL_LOAD_LOCK): two concurrent torch loads
        beside the acados build crashed the station twice on 2026-09-02."""
        from .upstream import MODEL_LOAD_LOCK
        with MODEL_LOAD_LOCK:
            self._load_impl(on_log)

    def _load_impl(self, on_log: Optional[Callable[[str], None]] = None) -> None:
        log = on_log or (lambda s: None)
        t_start = time.perf_counter()
        self.loading = True
        self.error = ""
        self.ready = False
        try:
            import dill
            import torch
            import torch.nn as nn
            import torchvision.transforms as T
            import diffusers
            import hydra
            import timm
            from omegaconf import OmegaConf
            from diffusers import DDIMScheduler

            ckpt = Path(self.ckpt)
            if not ckpt.is_file():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            if not Path(self.repo).is_dir():
                raise FileNotFoundError(f"UMI repo not found: {self.repo}")
            device = torch.device(self.device_str)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("device 'cuda' requested but CUDA is not "
                                   "available")

            log(f"DP policy: loading {ckpt.name} ({self.eval_transforms} "
                f"transforms, {self.num_inference_steps} DDIM steps)")
            Encoder, Denoiser = _import_upstream(self.repo)

            OmegaConf.register_new_resolver("eval", safe_eval, replace=True)
            with open(ckpt, "rb") as fh:
                payload = torch.load(fh, map_location="cpu",
                                     pickle_module=dill, weights_only=False)
            cfg = payload["cfg"]
            shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
            policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
            enc_cfg = dict(policy_cfg["obs_encoder"])
            sched_cfg = {k: v for k, v in policy_cfg["noise_scheduler"].items()
                         if k != "_target_"}
            target = policy_cfg.get("_target_", "")
            if not target.endswith("DiffusionTransformerTimmPolicy"):
                raise ValueError(f"this loader transcribes "
                                 f"DiffusionTransformerTimmPolicy; the "
                                 f"checkpoint's policy is {target!r}")
            if not str(policy_cfg["noise_scheduler"].get("_target_", "")) \
                    .endswith("DDIMScheduler"):
                raise ValueError("this loader transcribes the DDIM sampling "
                                 "loop; the checkpoint's scheduler is "
                                 f"{policy_cfg['noise_scheduler'].get('_target_')!r}")

            image_key, image_shape, lowdim_shapes, obs_keys, obs_horizon = \
                _obs_shapes_from_meta(shape_meta)
            action_meta = dict(shape_meta["action"])
            action_dim = int(action_meta["shape"][0])
            action_horizon = int(action_meta["horizon"])
            # The action CONTRACT (module docstring): declared name first,
            # width second, and never both disagreeing.
            action_repr = resolve_action_repr(action_meta, action_dim)
            yaw_tilt = action_meta.get("yaw_axis_cam_tilt_deg")
            yaw_tilt = None if yaw_tilt is None else float(yaw_tilt)
            yaw_R = _yaw_axis_R(action_meta.get("yaw_axis_R_frd_cam"))
            # The roll/pitch label convention of the 6-DoF variant (None on
            # every 5-dim / pose10d checkpoint; the backend's mount check
            # REQUIRES it for pos_rpy_width and compares it with
            # RPY_CONVENTION_STATION).
            rpy_conv = rpy_convention_of(action_meta)

            # -- transforms (module docstring) ---------------------------
            raw_tf = list(enc_cfg.get("transforms") or [])
            if not raw_tf or raw_tf[0].get("type") != "RandomCrop":
                raise ValueError("expected transforms[0] to be the RandomCrop "
                                 f"entry, got {raw_tf[:1]}")
            ratio = float(raw_tf[0]["ratio"])
            crop = int(image_shape[-1] * ratio)
            resize = T.Resize(size=image_shape[-1], antialias=True)
            if self.eval_transforms == "center":
                transforms = [T.CenterCrop(crop), resize]
            elif self.eval_transforms == "train":
                transforms = [T.RandomCrop(size=crop), resize] + [
                    hydra.utils.instantiate(e) for e in raw_tf[1:]]
            else:
                transforms = [nn.Identity()]
            tf_desc = [repr(t) for t in transforms]

            # -- encoder + denoiser ---------------------------------------
            # The trained encoder's gate, reproduced: upstream swaps BatchNorm for
            # GroupNorm only when ``use_group_norm and not pretrained``; the
            # 2026-09-10 ``force_group_norm`` flag (variant C) applies the swap
            # despite ``pretrained``. Getting this wrong is not subtle — a
            # GroupNorm checkpoint has no ``running_mean``/``running_var`` and
            # the strict load below refuses it.
            force_gn = bool(enc_cfg.get("force_group_norm", False))
            use_gn = bool(enc_cfg.get("use_group_norm", False)) \
                and (force_gn or not bool(enc_cfg.get("pretrained", False)))
            # ``key_transforms`` (2026-09-23, per-key train-time augmentation
            # such as ColorJitter on camera0_rgb only) is a training concern:
            # the station applies ``eval_transforms`` to every image key, so
            # the raw yaml dicts must not reach the encoder constructor.
            enc_kwargs = {k: v for k, v in enc_cfg.items()
                          if k not in ("_target_", "transforms", "shape_meta",
                                       "pretrained", "use_group_norm",
                                       "force_group_norm", "key_transforms")}
            encoder = Encoder(shape_meta=shape_meta, transforms=transforms,
                              pretrained=False, use_group_norm=use_gn,
                              **enc_kwargs)
            obs_shape = tuple(encoder.output_shape())            # (1, N, n_emb)
            n_emb = int(policy_cfg["n_emb"])
            if obs_shape[-1] != n_emb:
                raise ValueError(f"encoder emits {obs_shape[-1]}-d tokens, "
                                 f"policy expects n_emb {n_emb}")
            model = Denoiser(input_dim=action_dim, output_dim=action_dim,
                             action_horizon=action_horizon,
                             n_layer=int(policy_cfg["n_layer"]),
                             n_head=int(policy_cfg["n_head"]),
                             n_emb=n_emb,
                             max_cond_tokens=int(obs_shape[-2]) + 1,
                             p_drop_attn=float(policy_cfg["p_drop_attn"]))

            # -- weights: split three ways, strict ------------------------
            sd = _select_weights(payload["state_dicts"], self.weights, log)
            enc_sd = {k[len(_PFX_ENC):]: v for k, v in sd.items()
                      if k.startswith(_PFX_ENC)}
            model_sd = {k[len(_PFX_MODEL):]: v for k, v in sd.items()
                        if k.startswith(_PFX_MODEL)}
            # ``_dummy_variable`` is ModuleAttrMixin's device/dtype probe
            # on the policy object itself — not a weight.
            other = [k for k in sd if not (k.startswith(_PFX_ENC)
                                           or k.startswith(_PFX_MODEL)
                                           or k.startswith(_PFX_NORM)
                                           or k == "_dummy_variable")]
            if other:
                raise ValueError(f"{self.weights} has {len(other)} keys outside "
                                 f"obs_encoder./model./normalizer.: "
                                 f"{other[:5]}")
            encoder.load_state_dict(enc_sd, strict=True)
            model.load_state_dict(model_sd, strict=True)
            norm_np = _Normalizer.from_state_dict(sd, _PFX_NORM)
            for k in list(obs_keys) + ["action"]:
                if k not in norm_np.keys:
                    raise ValueError(f"normalizer has no entry for {k!r}")
            if norm_np.dim("action") != action_dim:
                raise ValueError("normalizer action dim "
                                 f"{norm_np.dim('action')} != {action_dim}")
            a_scale, a_offset = (np.asarray(v, float) for v in norm_np.params("action"))
            ident = identity_normaliser_columns(action_repr, a_scale, a_offset)
            if ident:
                raise ValueError(
                    f"{action_repr} checkpoint: action normalizer column(s) "
                    f"{ident} are IDENTITY (scale 1, offset 0) -- "
                    f"the angle column(s) were never range-normalised "
                    f"(upstream get_normalizer's pose10d slice); this "
                    f"checkpoint must not be flown")
            a_stats = norm_np.input_stats("action")
            action_range = (
                {"min": np.asarray(a_stats["min"], float).reshape(-1).tolist(),
                 "max": np.asarray(a_stats["max"], float).reshape(-1).tolist()}
                if "min" in a_stats and "max" in a_stats else None)
            if action_range is not None and (
                    len(action_range["min"]) != action_dim
                    or len(action_range["max"]) != action_dim):
                raise ValueError("normalizer action input_stats width != "
                                 f"action dim {action_dim}")
            pickles = payload.get("pickles") or {}
            ckpt_epoch = _pickled_int(pickles.get("epoch"), dill.loads)
            ckpt_global_step = _pickled_int(pickles.get("global_step"), dill.loads)

            encoder.eval()
            model.eval()
            encoder.to(device)
            model.to(device)
            for m in (encoder, model):
                for p in m.parameters():
                    p.requires_grad_(False)

            scheduler = DDIMScheduler(**sched_cfg)
            scheduler.set_timesteps(self.num_inference_steps)
            timesteps = [int(t) for t in scheduler.timesteps]

            # -- contract ------------------------------------------------
            dss = {k: int(a["down_sample_steps"])
                   for k, a in shape_meta["obs"].items()}
            dss["action"] = int(shape_meta["action"]["down_sample_steps"])
            if len(set(dss.values())) != 1:
                raise ValueError(f"mixed down_sample_steps {dss}")
            down_sample_steps = dss["action"]
            dataset_path = None
            try:
                dataset_path = str(cfg.task.dataset_path)
            except Exception:                                    # noqa: BLE001
                dataset_path = None
            zattrs = read_zip_zattrs(dataset_path) if dataset_path else None
            if zattrs is not None and "fps" in zattrs:
                fps = float(zattrs["fps"])
                fps_source = f"dataset zip .zattrs ({dataset_path})"
                if abs(fps - self.dataset_fps) > 1e-6:
                    raise ValueError(
                        f"dataset fps mismatch: the training zip says "
                        f"{fps} but policy.dataset_fps is {self.dataset_fps} "
                        f"— fix the config rather than guess obs_dt")
            else:
                fps = self.dataset_fps
                fps_source = (f"policy.dataset_fps arg (training zip "
                              f"{dataset_path!r} not readable)")
            obs_dt = down_sample_steps / fps
            contract = {
                "obs_keys": list(obs_keys),
                "image_key": image_key,
                "obs_horizon": int(obs_horizon),
                "action_horizon": action_horizon,
                "action_dim": action_dim,
                # The action CONTRACT (2026-09-07): which columns the (16, D)
                # output carries, the training range per column (normalizer
                # input_stats min/max -- with DDIM clip_sample the network
                # cannot leave it), the mount the yaw label was defined on
                # (None on a legacy pose10d checkpoint), and the epoch /
                # global step the file was saved at (provenance for the run
                # meta; None when the payload has no pickles).
                "action_repr": action_repr,
                "action_range": action_range,
                "yaw_axis_cam_tilt_deg": yaw_tilt,
                "yaw_axis_R_frd_cam": yaw_R,
                # 6-DoF variant (2026-09-26): the roll/pitch label convention
                # the checkpoint declares (None = not a pos_rpy_width run).
                "rpy_convention": rpy_conv,
                "ckpt_epoch": ckpt_epoch,
                "ckpt_global_step": ckpt_global_step,
                "down_sample_steps": down_sample_steps,
                "obs_dt_s": float(obs_dt),
                "fps": float(fps),
                "fps_source": fps_source,
                "image_shape": tuple(image_shape),
                "lowdim_shapes": {k: tuple(v) for k, v in lowdim_shapes.items()},
            }

            # -- publish ---------------------------------------------------
            with self._lock:
                self._torch = torch
                self._device = device
                self._stream = (torch.cuda.Stream(device=device)
                                if device.type == "cuda" else None)
                self._encoder = encoder
                self._model = model
                self._scheduler = scheduler
                self._norm_np = norm_np
                self._norm_t = norm_np.to_torch(torch, device, torch.float32)
                self._contract = contract
                stats = {}
                for k in ("robot0_eef_pos", "robot0_gripper_width"):
                    if k in norm_np.keys:
                        s = norm_np.input_stats(k)
                        stats[k] = {n: np.asarray(s[n]).tolist()
                                    for n in ("min", "max") if n in s}
                self._desc = {
                    "stub": False,
                    "ckpt": str(ckpt),
                    "ckpt_resolved": str(ckpt.resolve()),
                    "ckpt_sha1_head": sha1_head(ckpt),
                    "cfg_fingerprint": cfg_fingerprint(shape_meta, policy_cfg),
                    "policy_target": target,
                    "model_name": str(enc_cfg.get("model_name")),
                    "encoder_batchnorm_kept": not use_gn,
                    "weights": self.weights,
            "num_inference_steps": self.num_inference_steps,
                    "num_train_timesteps": int(sched_cfg["num_train_timesteps"]),
                    "timesteps": timesteps,
                    "scheduler": dict(sched_cfg),
                    "eval_transforms": self.eval_transforms,
                    "transforms": tf_desc,
                    "crop_px": crop,
                    "n_obs_tokens": int(obs_shape[-2]),
                    "device": str(device),
                    "torch": torch.__version__,
                    "diffusers": diffusers.__version__,
                    "timm": timm.__version__,
                    "normalizer_input_stats": stats,
                    "normalizer_image_identity": bool(
                        np.all(norm_np.params(image_key)[0] == 1.0)
                        and np.all(norm_np.params(image_key)[1] == 0.0)),
                    "dataset_path": dataset_path,
                }
                self.ready = True
            # warm-up (A7): the first live plan must not pay for lazy init
            self.warmup_ms = self._warmup()
            self.load_seconds = time.perf_counter() - t_start
            log(f"DP policy ready in {self.load_seconds:.1f} s "
                f"(warm-up {self.warmup_ms:.0f} ms, {obs_shape[-2]} tokens, "
                f"sha1 {self._desc['ckpt_sha1_head'][:12]}, "
                f"action {action_repr} (16, {action_dim}), epoch "
                f"{ckpt_epoch if ckpt_epoch is not None else '?'}, "
                f"obs_dt {obs_dt*1e3:.1f} ms from {fps_source.split(' (')[0]})")
        finally:
            self.loading = False

    def _warmup(self) -> float:
        c = self._contract
        obs = {c["image_key"]: np.zeros((c["obs_horizon"],) + tuple(c["image_shape"]),
                                        np.float32)}
        for k, shp in c["lowdim_shapes"].items():
            row = np.zeros(shp, np.float32)
            if shp == (6,):
                row[:] = ROT6D_IDENTITY
            elif shp == (1,):
                row[:] = STUB_WIDTH_M
            obs[k] = np.repeat(row[None], c["obs_horizon"], axis=0)
        _, ms = self.predict(obs, seed=0, _count=False)
        return ms

    def close(self) -> None:
        with self._lock:
            self.ready = False
            torch = self._torch
            self._encoder = None
            self._model = None
            self._scheduler = None
            self._norm_t = None
            self._stream = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- queries -----------------------------------------------------------
    @property
    def contract(self) -> dict:
        return dict(self._contract)

    @property
    def normalizer(self) -> Optional[_Normalizer]:
        """The numpy normalizer (tests / parity)."""
        return self._norm_np

    def _check_obs(self, obs: dict) -> dict:
        c = self._contract
        if not c:
            raise RuntimeError("policy session not loaded")
        out = {}
        for key in c["obs_keys"]:
            if key not in obs:
                raise KeyError(f"obs is missing {key!r}")
            v = np.ascontiguousarray(obs[key], dtype=np.float32)
            want = ((c["obs_horizon"],) + tuple(c["image_shape"])
                    if key == c["image_key"]
                    else (c["obs_horizon"],) + tuple(c["lowdim_shapes"][key]))
            if v.shape != want:
                raise ValueError(f"obs[{key!r}] has shape {v.shape}, "
                                 f"contract wants {want} (no batch dim)")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"obs[{key!r}] has non-finite values")
            out[key] = v
        return out

    def predict(self, obs: Dict[str, np.ndarray], *, seed: Optional[int] = None,
                _count: bool = True) -> Tuple[np.ndarray, float]:
        """Transcription of ``DiffusionTransformerTimmPolicy.predict_action``
        + ``conditional_sample`` for batch 1.

        ``obs``: arrays WITHOUT a batch dim (image (T, 3, H, W) float [0, 1],
        lowdim (T, D)). Returns (action (16, D) float32, infer_ms) with D =
        ``contract["action_dim"]`` in ``contract["action_repr"]`` (module
        docstring).
        ``seed`` fixes the initial-noise generator (parity / determinism
        tests); flight calls leave it None. The conditioning mask of
        ``predict_action`` is all-False by construction (``cond_data`` zeros,
        ``cond_mask`` zeros), so its two masked assignments are no-ops and
        are not replayed.
        """
        obs_np = self._check_obs(obs)
        with self._lock:
            if not self.ready or self._model is None:
                why = self.error or ("loading" if self.loading else "closed")
                raise RuntimeError(f"policy session not ready ({why})")
            torch = self._torch
            encoder, model, scheduler = self._encoder, self._model, self._scheduler
            assert not encoder.training and not model.training, \
                "policy modules must be in eval mode (dropout/BN off)"
            c = self._contract
            dev = self._device
            t0 = time.perf_counter()
            stream_ctx = (torch.cuda.stream(self._stream) if self._stream is not None
                          else _nullcontext())
            with torch.no_grad(), stream_ctx:
                obs_t = {k: torch.from_numpy(v)[None].to(dev)
                         for k, v in obs_np.items()}
                nobs = self._norm_t.normalize_dict(obs_t)
                tokens = encoder(nobs)                             # (1, N, n_emb)
                gen = None
                if seed is not None:
                    gen = torch.Generator(device=dev).manual_seed(int(seed))
                shape = (1, c["action_horizon"], c["action_dim"])
                trajectory = torch.randn(size=shape, dtype=torch.float32,
                                         device=dev, generator=gen)
                if scheduler.num_inference_steps != self.num_inference_steps:
                    scheduler.set_timesteps(self.num_inference_steps)
                for t in scheduler.timesteps:
                    model_output = model(trajectory, t, tokens)
                    trajectory = scheduler.step(model_output, t, trajectory,
                                                generator=gen).prev_sample
                action = self._norm_t.unnormalize("action", trajectory)
                out = action[0].float().cpu().numpy()
            if self._stream is not None:
                self._stream.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            if _count:
                self._infer_ms.append(ms)
                self._n_predict += 1
        out = np.ascontiguousarray(out, dtype=np.float32)
        if not np.all(np.isfinite(out)):
            raise ValueError("policy emitted a non-finite action")
        return out, float(ms)

    def describe(self) -> dict:
        d = dict(self._desc)
        d.update({
            "ready": bool(self.ready),
            "loading": bool(self.loading),
            "error": self.error,
            "load_s": float(self.load_seconds),
            "warmup_ms": float(self.warmup_ms),
            "n_predict": int(self._n_predict),
            "infer_ms_p50": (float(np.percentile(self._infer_ms, 50))
                             if self._infer_ms else float("nan")),
            "infer_ms_max": (float(max(self._infer_ms))
                             if self._infer_ms else float("nan")),
            "contract": dict(self._contract),
        })
        return d


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------
# the stub (A14): the same surface, a canned chunk, no torch
# --------------------------------------------------------------------------
class StubPolicySession:
    """Stand-in with :class:`DpPolicySession`'s surface and no network.

    ``predict`` returns a straight-ahead chunk: knot k at
    ``[0, 0, STUB_SPEED_M_S * k * obs_dt]`` in the TCP frame (TCP +z is
    forward in the camera optical frame), a yaw ramp of ``dyaw_rate_rad_s *
    k * obs_dt`` (0 by default: straight), width ``STUB_WIDTH_M`` — so
    demo_e2e drives the REAL PolicyWorker / MpcWorker path (clock conversion,
    epoch, estimator, filter, stitcher) with a plan whose geometry is known.
    ``action_repr`` picks the width: ``pos_yaw_width`` (default, the flown
    5-dim contract), ``pos_rpy_width`` (the 6-DoF variant, 2026-09-26:
    columns 0:4 and 6 exactly as the 5-dim chunk, roll/pitch columns 4:6
    identically 0 unless ``roll_ramp_deg`` / ``pitch_ramp_deg`` ramp them
    linearly from 0 at knot 0 to that value at the last knot -- the
    ``stub_rp`` spelling of :func:`make_policy_session` sets pitch 5 deg
    [예측]) or the legacy ``pose10d`` (identity rot6d in columns 3:9, width
    in 9) for the recomposition tools. The obs contract is the trained
    policy's (shape_meta of the 2026-09-01 / 2026-09-07 checkpoints,
    identical) so a worker that satisfies the stub satisfies the real
    session; obs shapes are checked the same way, and the contract carries
    the same keys (``action_range`` is STUB_ACTION_RANGE, the mount and
    epoch keys are None as on a legacy checkpoint; ``rpy_convention`` is
    the station constant on the 7-dim stub, None otherwise -- the mount
    check needs it to let a 7-dim stub arm).
    """

    OBS_KEYS = ("camera0_depth", "robot0_eef_pos", "robot0_eef_rot_axis_angle",
                "robot0_gripper_width", "robot0_eef_rot_axis_angle_wrt_start")
    LOWDIM_SHAPES = {"robot0_eef_pos": (3,), "robot0_eef_rot_axis_angle": (6,),
                     "robot0_gripper_width": (1,),
                     "robot0_eef_rot_axis_angle_wrt_start": (6,)}

    def __init__(self, ckpt="stub", repo="", *, num_inference_steps: int = 16,
                 eval_transforms: str = "center", device: str = "cpu",
                 dataset_fps: float = 30.0, weights: str = "ema_model",
                 down_sample_steps: int = 2,
                 action_horizon: int = 16, obs_horizon: int = 2,
                 image_shape=(3, 224, 224),
                 action_repr: str = ACTION_REPR_POS_YAW_WIDTH,
                 dyaw_rate_rad_s: float = 0.0,
                 roll_ramp_deg: float = 0.0, pitch_ramp_deg: float = 0.0):
        self.ckpt, self.repo = str(ckpt), str(repo)
        self.num_inference_steps = int(num_inference_steps)
        self.eval_transforms = str(eval_transforms)
        self.weights = str(weights)
        self.device_str = str(device)
        self.dataset_fps = float(dataset_fps)
        if self.dataset_fps <= 0.0 or not math.isfinite(self.dataset_fps):
            raise ValueError("dataset_fps must be positive")
        if action_repr not in ACTION_DIM_BY_REPR:
            raise ValueError(f"action_repr must be one of "
                             f"{sorted(ACTION_DIM_BY_REPR)}, got {action_repr!r}")
        self.action_repr = str(action_repr)
        self.dyaw_rate_rad_s = float(dyaw_rate_rad_s)
        if not math.isfinite(self.dyaw_rate_rad_s):
            raise ValueError("dyaw_rate_rad_s must be finite")
        self.roll_ramp_deg = float(roll_ramp_deg)
        self.pitch_ramp_deg = float(pitch_ramp_deg)
        if not (math.isfinite(self.roll_ramp_deg) and math.isfinite(self.pitch_ramp_deg)):
            raise ValueError("roll_ramp_deg / pitch_ramp_deg must be finite")
        if (self.roll_ramp_deg or self.pitch_ramp_deg) \
                and self.action_repr != ACTION_REPR_POS_RPY_WIDTH:
            raise ValueError("roll_ramp_deg / pitch_ramp_deg need action_repr "
                             f"{ACTION_REPR_POS_RPY_WIDTH!r} (no roll/pitch column "
                             f"in {self.action_repr!r})")
        # The 7-dim stub carries the C3 mount and the convention the REAL
        # 7-dim checkpoint would (the backend's mount check REQUIRES both for
        # pos_rpy_width); the 5-dim / pose10d stub keeps None (legacy shape).
        if self.action_repr == ACTION_REPR_POS_RPY_WIDTH:
            from ..control.policy_frames import R_BT_C3, YAW_AXIS_CAM_TILT_DEG
            yaw_tilt: Optional[float] = float(YAW_AXIS_CAM_TILT_DEG)
            yaw_R: Optional[list] = np.asarray(R_BT_C3, float).reshape(3, 3).tolist()
            rpy_conv: Optional[dict] = {
                k: (list(v) if isinstance(v, list) else v)
                for k, v in RPY_CONVENTION_STATION.items()}
        else:
            yaw_tilt, yaw_R, rpy_conv = None, None, None
        self.ready = False
        self.loading = False
        self.error = ""
        self.load_seconds = 0.0
        self.warmup_ms = 0.0
        self._n_predict = 0
        self._infer_ms = deque(maxlen=_INFER_HISTORY)
        obs_dt = int(down_sample_steps) / self.dataset_fps
        self._contract = {
            "obs_keys": list(self.OBS_KEYS),
            "image_key": "camera0_depth",
            "obs_horizon": int(obs_horizon),
            "action_horizon": int(action_horizon),
            "action_dim": int(ACTION_DIM_BY_REPR[self.action_repr]),
            "action_repr": self.action_repr,
            "action_range": {"min": list(STUB_ACTION_RANGE[self.action_repr][0]),
                             "max": list(STUB_ACTION_RANGE[self.action_repr][1])},
            "yaw_axis_cam_tilt_deg": yaw_tilt,
            "yaw_axis_R_frd_cam": yaw_R,
            "rpy_convention": rpy_conv,
            "ckpt_epoch": None,
            "ckpt_global_step": None,
            "down_sample_steps": int(down_sample_steps),
            "obs_dt_s": float(obs_dt),
            "fps": self.dataset_fps,
            "fps_source": "stub (policy.dataset_fps)",
            "image_shape": tuple(int(s) for s in image_shape),
            "lowdim_shapes": dict(self.LOWDIM_SHAPES),
        }

    def start_async(self, on_log=None) -> None:
        self.load(on_log)

    def load(self, on_log=None) -> None:
        self.ready = True
        self.loading = False
        if on_log:
            on_log("DP policy STUB ready (canned straight-ahead chunk, "
                   f"{STUB_SPEED_M_S} m/s along TCP +z, dyaw "
                   f"{self.dyaw_rate_rad_s:g} rad/s, width {STUB_WIDTH_M} m, "
                   f"{self.action_repr} (16, {self._contract['action_dim']})"
                   + (f", roll/pitch ramp {self.roll_ramp_deg:g}/"
                      f"{self.pitch_ramp_deg:g} deg at the last knot"
                      if self.action_repr == ACTION_REPR_POS_RPY_WIDTH else "")
                   + ")")

    def close(self) -> None:
        self.ready = False

    @property
    def contract(self) -> dict:
        return dict(self._contract)

    _check_obs = DpPolicySession._check_obs

    def predict(self, obs: Dict[str, np.ndarray], *, seed=None,
                _count: bool = True) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        self._check_obs(obs)
        if not self.ready:
            raise RuntimeError("stub policy session closed")
        c = self._contract
        K = c["action_horizon"]
        out = np.zeros((K, c["action_dim"]), np.float32)
        tk = np.arange(K) * c["obs_dt_s"]
        out[:, 2] = STUB_SPEED_M_S * tk
        if self.action_repr == ACTION_REPR_POS_YAW_WIDTH:
            out[:, 3] = self.dyaw_rate_rad_s * tk
            out[:, 4] = STUB_WIDTH_M
        elif self.action_repr == ACTION_REPR_POS_RPY_WIDTH:
            # cols 0:4 and 6 exactly as the 5-dim chunk; roll/pitch (4:6) a
            # linear ramp from 0 at knot 0 to *_ramp_deg at the last knot
            # (0 = level: the dropped-and-logged wiring case).
            out[:, 3] = self.dyaw_rate_rad_s * tk
            frac = np.arange(K, dtype=np.float64) / max(K - 1, 1)
            out[:, 4] = math.radians(self.roll_ramp_deg) * frac
            out[:, 5] = math.radians(self.pitch_ramp_deg) * frac
            out[:, 6] = STUB_WIDTH_M
        else:                                    # legacy pose10d
            out[:, 3:9] = ROT6D_IDENTITY
            out[:, 9] = STUB_WIDTH_M
        ms = (time.perf_counter() - t0) * 1e3
        if _count:
            self._n_predict += 1
            self._infer_ms.append(ms)
        return out, float(ms)

    def describe(self) -> dict:
        return {
            "stub": True,
            "ckpt": self.ckpt,
            "ckpt_sha1_head": "",
            "cfg_fingerprint": "",
            "model_name": "stub",
            "weights": self.weights,
            "num_inference_steps": self.num_inference_steps,
            "timesteps": [],
            "eval_transforms": self.eval_transforms,
            "action_repr": self.action_repr,
            "stub_speed_m_s": STUB_SPEED_M_S,
            "stub_dyaw_rate_rad_s": self.dyaw_rate_rad_s,
            "stub_rp_ramp_deg": (self.roll_ramp_deg, self.pitch_ramp_deg),
            "stub_width_m": STUB_WIDTH_M,
            "ready": bool(self.ready),
            "loading": False,
            "error": self.error,
            "load_s": 0.0,
            "warmup_ms": 0.0,
            "n_predict": int(self._n_predict),
            "infer_ms_p50": (float(np.percentile(self._infer_ms, 50))
                             if self._infer_ms else float("nan")),
            "infer_ms_max": (float(max(self._infer_ms))
                             if self._infer_ms else float("nan")),
            "contract": dict(self._contract),
        }


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def make_policy_session(cfg_block: dict, opts=None, *, ckpt=None):
    """Build the session the ``policy:`` config block (plus the overrides
    ``opts.policy_ckpt`` / ``--policy-repo`` on ``opts``) asks for.

    Precedence for the checkpoint: the explicit ``ckpt`` kwarg (the panel's
    picker, via ``PolicyWorker.set_ckpt`` — 2026-09-11, operator request:
    the ``--policy-ckpt`` flag is gone and the file is chosen in the GUI),
    then ``opts.policy_ckpt`` (still honoured when a test/tool sets it
    directly on its Opts class: the stub sessions), then the block. With
    ``ckpt=None`` the pre-2026-09-11 precedence is unchanged.

    ``ckpt == "stub"`` (any source) picks :class:`StubPolicySession`; the
    other spellings in ``STUB_CKPT_VARIANTS`` (``stub7`` = 7-dim level,
    ``stub_rp`` = 7-dim with a 5 deg pitch ramp, 2026-09-26) pick the stub
    with those kwargs. Anything else is a checkpoint path for
    :class:`DpPolicySession`. The real session is NOT loaded here — the
    caller decides the thread (``start_async``) — but its constructor
    already validates the ``eval_transforms`` / ``num_inference_steps``
    choices.
    """
    cfg_block = dict(cfg_block or {})
    ckpt = (ckpt or getattr(opts, "policy_ckpt", None)
            or cfg_block.get("ckpt", DEFAULT_CKPT))
    repo = getattr(opts, "policy_repo", None) or cfg_block.get("repo", DEFAULT_UMI_REPO)
    weights = (getattr(opts, "policy_weights", None)
               or cfg_block.get("weights", "ema_model"))
    kw = dict(num_inference_steps=int(cfg_block.get("num_inference_steps", 16)),
              eval_transforms=str(cfg_block.get("eval_transforms", "center")),
              dataset_fps=float(cfg_block.get("dataset_fps", 30.0)),
              weights=str(weights))
    key = str(ckpt).strip().lower()
    if key in STUB_CKPT_VARIANTS:
        return StubPolicySession(key, repo, **kw, **STUB_CKPT_VARIANTS[key])
    return DpPolicySession(ckpt, repo, device=str(cfg_block.get("device", "cuda")),
                           **kw)
