#!/usr/bin/env python3
"""
policy.py — the diffusion-policy worker (``--policy``): depth + proprio in,
raw action chunks out, on its own thread.

    depth producer ──PolicyMailbox──▶ PolicyWorker ──bus.policy_plan──▶ MpcWorker
    MpcWorker ─────bus.policy_state─▶ PolicyWorker ──bus.policy_status─▶ MpcWorker + window

One worker, built by BOTH ``HardwareBackend`` and ``DemoBackend`` when
``--policy`` is given (spec DP_LIVE_POLICY_SPEC_V2 A14: the demo drives the
REAL worker with a stub network so the clock conversion, the epoch rule, the
pairing and the estimator are exercised offline). It owns:

* the network session (``perception/dp_policy.py``: the trained
  ``DiffusionTransformerTimmPolicy`` on the GPU, or ``StubPolicySession`` for
  ``ckpt: stub``), loaded asynchronously the moment the worker starts —
  ``--policy`` is the operator saying this run is about the policy, the same
  rule as ``--fstereo``;
* the depth observation builder (``perception/policy_obs.py``): the producer
  declares which GRID its millimetres live on (rect_left = FoundationStereo's
  host rectified-left map, color_aligned = the device's CAM_A-aligned depth,
  identity = the demo's 640x400) and the builder re-projects onto the
  training target grid and applies the training recipe bit for bit;
* the proprio history keyed on the FIX stamp (A6: one row per tag fix, never
  per tick — the assembler holds a fix across ticks, and rows appended per
  tick would quantise the 66.7 ms motion cue to {0, 1, 2} fixes);
* the depth ring (last 12 frames with their capture stamps) and the PAIRING
  rule (A18) that picks the two frames ``obs_dt`` apart;
* the status the controller's refusal reads (A13: stamped, >= 1 Hz when
  idle; a silent worker must not stay "ready").

What it deliberately does NOT do: it never composes a plan into NED, never
converts a clock, never touches the jaw. ``PolicyPlan.obs_t`` is on the
monotonic ``state.now()`` clock like every ``t_capture``; the ONE place that
crosses into the mission clock is ``MpcWorker._tick_policy_intake`` (A1).
And it only infers while ``PolicyState.active`` (the mission is armed and
running) — the GPU idles otherwise, and a plan built from a stale state would
be dropped by the consumer's epoch rule anyway (A8).

Timing (A18 / A19): the forward runs on the session's own CUDA stream, which
keeps it off FoundationStereo's QUEUE but not off its GPU — with the stereo
network running back to back beside it the same forward takes 152.8 ms p50
instead of 19.3 ms [측정: data/20260930/0930_220212/diag/
policy_vs_fstereo_contention.json, offline]. ``--policy-fs-schedule`` (see
perception/fs_gate.py and ``fs_gate`` below) is the switch for that; its
default, ``free``, is this paragraph unchanged. The worker tick is
20 ms and the inference period ``policy.period_s`` (0.5 s ≈ n_action_steps ×
obs_dt). Inference fires on depth-frame ARRIVAL once the period has elapsed —
not on the period timer's own phase, which would add a mean half frame
interval (69 ms at the measured 7.26 Hz [측정: rov_gui/tools/fstereo_bench_out/
policy_bench_20260902_141224.json]) to an intake budget that has none to
spare (verify 2026-09-02) — and ``_last_infer`` is stamped only when a
forward actually ran, so a skipped attempt retries on the next frame instead
of burning a whole period. (What the code does: a frame that arrived WHILE the
period was still running leaves the arrival flag set, so the first tick after
the period fires on the newest frame already in the ring; it waits for an
arrival only when none came since the last attempt.) A plan is emitted with EVERY field the record
needs to attribute it afterwards (the two proprio row stamps — taken at the
SAME spacing as the two depth frames, so image and proprio baselines agree —
the bracketing fix stamps, the real depth-pair spacing, whether the newest
frame was duplicated, the obs coverage and validity, the checkpoint hash, the
checkpoint's obs stride).
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..qt import Slot
from ..perception.dp_policy import RPY_CONVENTION_STATION, is_stub_ckpt
from ..perception.fs_gate import BURST_FRAMES
from ..state import (ACTION_REPR_POS_RPY_WIDTH, Conn, POLICY_CKPT_DEFAULT,
                     POLICY_UMI_REPO, PolicyPlan, PolicyStatus, SensorStat, now)
from .base import TimerWorker

REPO = Path(__file__).resolve().parents[2]

#: The obs contract's image key (spec v1 §0). The lowdim keys come from
#: ``policy_frames.lowdim_obs``.
IMAGE_KEY = "camera0_depth"

#: The training target grid (raw CAM_B 640x400) the builder warps onto.
WARP_SIZE = (640, 400)

#: Pairing tolerances (v2 A18), as fractions of obs_dt. PAIR_FALLBACK was
#: 2.5 (166.7 ms); at the measured FoundationStereo pair rate under the
#: --policy defaults — 7.26 Hz, solve 137 ms, panel latency 243 ms [측정:
#: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_141224.json, C3 실기,
#: alpha 0.5, scale 1.0, iters 16, 330 frames, GPU shared with a concurrent
#: test run] — frames are 138 ms apart, so the 'near' window (±40 ms around
#: t_d − 66.7 ms) is never hit and every pair is a fallback 29 ms under the
#: old cap: one jittered frame meant skip_pair. 3.0 (200 ms) keeps a single
#: slow solve from dropping the plan; the real spacing is recorded as
#: ``obs_pair_dt_s`` on every plan (verify 2026-09-02).
PAIR_TOL = 0.6
PAIR_FALLBACK = 3.0

#: A depth stamp newer than the newest tag fix by more than this fraction of
#: obs_dt is not observed but extrapolated: the proprio 'now' row would be
#: clamped to the last fix while the depth is later, so the motion cue and
#: the image disagree. Skip (``skip_fix_lag``) rather than under-scale the
#: cue silently (verify 2026-09-02).
FIX_LAG_TOL = 0.5

#: A PolicyState older than this is treated as INACTIVE (v2 A13): the
#: controller emits one per 20 Hz tick, so a second of silence is a dead or
#: stalled controller, and inferring against it would produce plans nobody
#: consumes.
STATE_STALE_S = 1.0

#: `--policy-fs-schedule only`: how long before the period ends the worker
#: asks FoundationStereo for its two frames, so the attempt lands on the
#: period instead of a burst later. [유도: one camera interval to wait for a
#: fresh pair (0-66.7 ms at 15 fps) + two frames at 75.8 ms each (data/
#: 20260930/0930_220212/diag/policy_vs_fstereo_contention.json) = 152-218 ms.]
#: A starting value only — the worker tracks the burst time it measures,
#: inside [ONLY_LEAD_MIN_S, ONLY_LEAD_MAX_S].
ONLY_LEAD_S = 0.2
ONLY_LEAD_MIN_S = 0.1
ONLY_LEAD_MAX_S = 0.4

#: The device-depth path's --depth-scale correction (C3VideoWorker applies it
#: at the source). The builder cannot know it; the record must (v2 A16).
DEVICE_DEPTH_SCALE_DEFAULT = 0.64

#: The handheld gripper's TCP offset in its camera frame, for the meta block's
#: comparison only (v2 A3). Duplicated from control/geometry so this module
#: stays importable without the control stack; the test asserts equality.
HANDHELD_TCP_OFFSET_CAM_M = (0.0355, 0.1293, 0.3186)

#: FALLBACK ONLY: the ``policy:`` defaults of spec v2 §B, used when the
#: loaded MpcConfig has no ``policy`` attribute (control/geometry.py is
#: owned by another agent and may lag). geometry.default_policy_block is
#: the single source of truth whenever it exists; ``meta()`` records which
#: one was used (``policy_block_source``).
_FALLBACK_POLICY_BLOCK = {
    "ckpt": POLICY_CKPT_DEFAULT,           # ONE source: rov_gui/state.py
    "repo": POLICY_UMI_REPO,
    # The DEPLOYED state dict (geometry.default_policy_block mirrors it): the
    # EMA copy carries the BatchNorm defect and is much worse [측정 2026-09-07,
    # 219-window held-out, same epoch 195: ema 33.0 mm vs model 17.3 mm].
    "weights": "model",
    "target_model": "configs/target_camera_underwater.yaml",
    "dataset_fps": 30.0, "num_inference_steps": 8, "eval_transforms": "center",
    "period_s": 0.5, "knot_dt_s": 0.2, "obs_max_age_s": 0.6, "stale_s": 1.5,
    # [결정: operator 2026-09-11 — 120 -> 500 s]; geometry.default_policy_block
    # and both YAMLs carry the same value (test_policy_ckpt_paths pins it).
    "max_run_s": 500.0,
    "anchor": "leash", "anchor_leash_m": 0.05, "anchor_leash_yaw_deg": 10.0,
    "v_max_m_s": 0.08, "a_max_m_s2": 0.20, "r_max_rad_s": 0.50,
    "anchor_max_m": 0.15, "jump_max_m": 0.06, "yaw_jump_max_deg": 8.0,
    "div_max_m": 0.25,
    "blend_s": 0.4, "blend_v_max_m_s": 0.16, "blend_a_max_m_s2": 1.0,
    "hold_tail": "mask", "hold_tail_taper_s": 0.0,
    "workspace_box_ned": [[-1.0, -1.0, -0.5], [1.0, 1.0, 0.5]],
    # 2026-09-08: jaw anchored to the C3 lens (cam_t_flu + [0.196, 0, -0.275]);
    # mirrors geometry.default_policy_block — re-derive both if cam_t_flu moves.
    "tcp_body_flu_m": [0.502, 0.0, -0.17], "tcp_offset_cam_m": None,
    "obs_view_forward_m": None,
    "min_obs_coverage": 0.985, "z_near_m": 0.20, "z_far_m": 3.00, "obs_res": 224,
    "gripper": False, "gripper_close_below": 0.30, "gripper_open_above": 0.67,
    "gripper_hold_max_s": 4.0,
    "gripper_width_open_m": 0.069, "gripper_width_closed_m": 0.042,
    "gripper_width_init_m": 0.069, "gripper_travel_s": 2.0,
}


# =============================================================================
# helpers shared with the producers and the CLI
# =============================================================================
def rect_left_grid_from_rig(rig):
    """The ``RectLeftGrid`` descriptor for a ``host_depth.StereoRig``.

    ONE place builds it (v2 A16): the FoundationStereo producer calls this to
    declare the mailbox grid, the worker reads that same descriptor back and
    hands it to the builder, so the fingerprint the mailbox gates on and the
    fingerprint the builder checks are the same object. ``K_live``/``D_live``
    (the raw CAM_B intrinsics at mono size) ride in ``rig.provenance`` when
    ``C3VideoWorker._build_rig`` stashed them; without them the builder's
    model check is SKIPPED and says so in the meta — never silently passed.
    """
    from ..perception.policy_obs import GridError, RectLeftGrid

    if getattr(rig, "R1", None) is None or getattr(rig, "P1", None) is None:
        # A rig built without stereoRectify output (a loaded dataset rig, a
        # constructor that only carried fx/baseline) has no rectified-left
        # geometry to un-rectify: refuse by name instead of handing the
        # builder a descriptor of None arrays that fails somewhere deeper.
        raise GridError("rect_left grid needs the rig's R1/P1 (stereoRectify "
                        "output); this StereoRig carries none — the policy "
                        "cannot map its depth onto the training grid")
    prov = dict(getattr(rig, "provenance", {}) or {})
    K_live = prov.get("left_K_live")
    D_live = prov.get("left_D_live")
    # The maps/arrays are not JSON and not needed: keep the scalar provenance.
    prov_small = {k: v for k, v in prov.items()
                  if k not in ("R1", "P1", "P2", "left_K_live", "left_D_live")}
    return RectLeftGrid(R1=np.asarray(rig.R1, float), P1=np.asarray(rig.P1, float),
                        mono_size=tuple(rig.mono_size), alpha=float(rig.alpha or 0.0),
                        K_live=None if K_live is None else np.asarray(K_live, float),
                        D_live=None if D_live is None else np.asarray(D_live, float),
                        provenance=prov_small)


def hydra_config_for(ckpt) -> Path | None:
    """The ``.hydra/config.yaml`` beside a training checkpoint, or None.

    The checkpoint itself is a torch pickle and cannot be read without torch;
    hydra writes the resolved training config next to it
    (``<run>/checkpoints/<name>.ckpt`` -> ``<run>/.hydra/config.yaml``), which
    is what a torch-free launch check can read.

    ``data/checkpoints/*.ckpt`` are SYMLINKS into their run folder (the
    2026-09-14 layout; the .json beside each says so), and the link's own
    folder has no .hydra — so the target is searched first. Until 2026-09-30
    it was not, and every launch from data/checkpoints skipped the FS parity
    check: run meta ``policy.worker.fstereo.training.status`` is
    ``no_hydra_config`` on all 178 non-stub blocks from 2026-09-14 on, against
    245 ``ok`` before [측정: data/2026*/*/*.meta.json, counted 2026-09-30].
    """
    p = Path(str(ckpt)).expanduser()
    try:
        bases = dict.fromkeys((p.resolve(), p))
    except (OSError, RuntimeError):
        bases = (p,)
    for base in bases:
        for parent in (base.parent.parent, base.parent):
            cand = parent / ".hydra" / "config.yaml"
            if cand.is_file():
                return cand
    return None


def training_depth_source(ckpt) -> dict:
    """What FoundationStereo settings the TRAINING depth store was built with.

    Reads ``task.dataset_path`` from the hydra config beside ``ckpt`` and the
    zip's ``.zattrs['depth_source']`` (zipfile only — no zarr, no torch).
    Returns a dict with ``status`` in {"ok", "stub", "no_hydra_config",
    "no_dataset_path", "no_zip", "no_depth_source", "error"} plus, when ok,
    ``iters``, ``scale``, ``checkpoint``, ``depth_scale_applied``,
    ``dataset_path`` and ``hydra_config``. Used by ``__main__.check_policy``
    (v2 A17) and recorded in the policy meta.
    """
    out: dict = {"status": "", "ckpt": str(ckpt)}
    if is_stub_ckpt(ckpt):
        out["status"] = "stub"
        return out
    cfg_path = hydra_config_for(ckpt)
    if cfg_path is None:
        out["status"] = "no_hydra_config"
        return out
    out["hydra_config"] = str(cfg_path)
    try:
        import yaml

        with open(cfg_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        ds = (cfg.get("task") or {}).get("dataset_path")
        if not ds:
            out["status"] = "no_dataset_path"
            return out
        out["dataset_path"] = str(ds)
        if not Path(str(ds)).is_file():
            out["status"] = "no_zip"
            return out
        from ..perception.dp_policy import read_zip_zattrs

        z = read_zip_zattrs(ds) or {}
        src = z.get("depth_source")
        if not isinstance(src, dict):
            out["status"] = "no_depth_source"
            return out
        out.update({
            "status": "ok",
            "iters": src.get("iters"),
            "scale": src.get("scale"),
            "checkpoint": src.get("checkpoint"),
            "depth_scale_applied": src.get("depth_scale_applied"),
            "tool": src.get("tool"),
            "fps": z.get("fps"),
        })
    except Exception as e:                                       # noqa: BLE001
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def fs_settings_mismatch(opts, src: dict) -> list[str]:
    """The FoundationStereo settings this run would fly with, against the
    training store's (v2 A17). Returns the mismatches as sentences (empty =
    parity). Only meaningful when ``src['status'] == 'ok'``."""
    bad: list[str] = []
    if src.get("status") != "ok":
        return bad
    iters = getattr(opts, "fstereo_iters", None)
    scale = getattr(opts, "fstereo_scale", None)
    size = getattr(opts, "fstereo_size", None)
    if src.get("iters") is not None and iters is not None \
            and int(iters) != int(src["iters"]):
        bad.append(f"--fstereo-iters {int(iters)} vs training {int(src['iters'])}")
    if size:
        bad.append(f"--fstereo-size {size} vs training scale {src.get('scale')} "
                   f"(full 640x400)")
    elif src.get("scale") is not None and scale is not None \
            and abs(float(scale) - float(src["scale"])) > 1e-9:
        bad.append(f"--fstereo-scale {float(scale):g} vs training "
                   f"{float(src['scale']):g}")
    ck = effective_fstereo_ckpt(opts)
    if ck is not None and src.get("checkpoint") and \
            not same_fstereo_ckpt(ck, src["checkpoint"]):
        typed = bool(getattr(opts, "fstereo_ckpt", None))
        bad.append(f"{'--fstereo-ckpt' if typed else 'default fstereo ckpt'} "
                   f"{ck} vs training {src['checkpoint']}")
    return bad


def same_fstereo_ckpt(a, b) -> bool:
    """Are ``a`` and ``b`` the same FoundationStereo checkpoint, wherever each
    one lives?

    The same resolved path; else, both files present, the same SHA-1 of the
    first 8 MiB (the digest every run records as ``ckpt_sha1_first_8mib``);
    else, one of them gone, the survivor's digest against upstream's
    published one for the gone file's model id ``<run>/<file>``
    (``fstereo.UPSTREAM_CKPT_SHA1``) — an id it does not know cannot be
    verified and counts as different. Only when nothing can be read is the
    model id alone compared.

    A path is not an identity: on 2026-09-30 the checkout moved into this
    repo (external/FoundationStereo), and the training stores name
    ~/Desktop/FoundationStereo/... (deleted) or ~/Desktop/data collection/
    FoundationStereo/... (a byte-identical copy), so a path compare reads the
    same weights as another model. Nor may a path this user cannot stat
    (EACCES, ENAMETOOLONG, ``~other``) raise: this runs in the launch check
    and at a panel pick, where the old path compare only ever warned. It is
    treated like a missing file.
    """
    from ..perception.fstereo import UPSTREAM_CKPT_SHA1, _sha1_head

    def probe(x):
        try:
            p = Path(str(x)).expanduser()
            return p, p.is_file()
        except (OSError, RuntimeError, ValueError):
            return Path(str(x)), False

    (pa, ha), (pb, hb) = probe(a), probe(b)
    try:
        if pa.resolve() == pb.resolve():
            return True
    except (OSError, RuntimeError):
        pass
    try:
        if ha and hb:
            return _sha1_head(pa) == _sha1_head(pb)
        if ha or hb:
            have, gone = (pa, pb) if ha else (pb, pa)
            want = UPSTREAM_CKPT_SHA1.get(f"{gone.parent.name}/{gone.name}")
            return want is not None and _sha1_head(have) == want
    except OSError:
        pass                            # unreadable: only the model id is left
    return (pa.parent.name, pa.name) == (pb.parent.name, pb.name)


def effective_fstereo_ckpt(opts):
    """The FoundationStereo checkpoint this run would actually load: the typed
    ``--fstereo-ckpt``, else ``FStereoSession``'s default (``--fstereo-repo`` /
    ``FOUNDATION_STEREO_REPO`` / DEFAULT_REPO, joined with DEFAULT_CKPT). The
    parity check must compare THIS, not only a typed value: a training store
    built from another checkpoint used to pass silently whenever the flag was
    left at its default (verify 2026-09-02)."""
    import os

    ck = getattr(opts, "fstereo_ckpt", None)
    if ck:
        return Path(str(ck)).expanduser()
    try:
        from ..perception.fstereo import DEFAULT_CKPT, DEFAULT_REPO
    except Exception:                                            # noqa: BLE001
        return None
    repo = (getattr(opts, "fstereo_repo", None)
            or os.environ.get("FOUNDATION_STEREO_REPO") or DEFAULT_REPO)
    return Path(str(repo)).expanduser() / DEFAULT_CKPT


def _rows_for(hist, t_now: float, spacing: float):
    """``EtaHistory.rows_for`` (control/policy_frames.py, the frozen contract:
    ``(eta_prev, eta_now, info)`` with ``t_now_eff = min(t_now, latest_t)``,
    ``t_prev = t_now_eff - spacing``, ``info = {t_prev, t_now, fix_lag_s,
    degenerate}``). The method is owned by the frames module; until it lands
    there this shim builds the SAME answer from the public API (``latest_t``,
    ``interp``, ``fix_stamps``, ``span``) so the worker and its tests do not
    wait on it — and drops out the moment the method exists."""
    fn = getattr(hist, "rows_for", None)
    if fn is not None:
        return fn(float(t_now), float(spacing))
    latest = hist.latest_t
    if latest is None:
        raise ValueError("EtaHistory is empty")
    t_now = float(t_now)
    spacing = float(spacing)
    t_eff = min(t_now, float(latest))
    t_prev = t_eff - spacing
    fp, fn_ = hist.fix_stamps(t_prev), hist.fix_stamps(t_eff)
    same_single = (len(fp) == 1 and len(fn_) == 1 and fp[0] == fn_[0])
    info = {"t_prev": t_prev, "t_now": t_eff,
            "fix_lag_s": max(0.0, t_now - float(latest)),
            "degenerate": bool(same_single or hist.span() < spacing)}
    return hist.interp(t_prev), hist.interp(t_eff), info


# =============================================================================
# the worker
# =============================================================================
class PolicyWorker(TimerWorker):
    """See the module docstring. Constructed by the backend with the shared
    ``PolicyMailbox``; ``cfg``/``nav_cfg``/``session_factory`` exist for the
    offline tests (the defaults load hw_mpc.yaml / hw_nav.yaml like
    ``TagNavWorker`` and pick the session from the config block)."""

    RING = 12

    def __init__(self, bus, mailbox, opts, *, cfg=None, nav_cfg=None,
                 session_factory=None):
        super().__init__("dp-policy", interval_ms=20)
        self.bus = bus
        self.mailbox = mailbox
        self.opts = opts
        self.cfg = cfg
        self.nav_cfg = nav_cfg
        self._session_factory = session_factory
        self.session = None
        self.builder = None
        self.hist = None                       # policy_frames.EtaHistory
        self.pc: dict = {}                     # the policy block
        self.policy_block_source = ""
        self.T_bt = None                       # (4,4) T_body_tcp (body FRD)
        self.tcp_offset_cam_m = None
        self.t_body_tcp = None
        #: hw_nav's R_frd_cam('main'), kept for the MOUNT check against the
        #: checkpoint's yaw-axis rotation (_check_mount, 2026-09-07).
        self._R_bc = None
        self.obs_view_shift_cam_m = None       # policy.obs_view_forward_m in the camera frame (setup)
        self._mount_ok = True
        self._mount_why = ""
        self._mount_said = False
        self.enabled = False
        self.fstereo_meta_fn = None            # injected by HardwareBackend
        #: perception.fs_gate.FsGate, injected by HardwareBackend when
        #: --policy-fs-schedule is `yield` or `only` and a FoundationStereo
        #: worker exists. None — the default `free`, the demo source, device
        #: depth, every offline tool — makes NO call into the gate: the
        #: pre-2026-10-01 behaviour, unchanged.
        self.fs_gate = None
        #: `only`: the burst this worker asked FoundationStereo for and is
        #: waiting on ({"t": request stamp}), None between bursts.
        self._burst = None
        #: `only`: how long before the period ends the burst is requested, so
        #: the attempt lands on the period. Tracks the measured burst time.
        self._only_lead_s = ONLY_LEAD_S
        self._sched_n = {"bursts": 0, "burst_timeouts": 0, "burst_refused": 0}
        self._burst_warn_t = -1e9
        #: The CONTROLLER's answer to "which folder is this run". Injected by
        #: the backend as ``MpcWorker._run_dir``; see the --record-depth block
        #: in setup() for why the recorder must not resolve its own.
        self.run_dir_fn = None
        #: bus.LatestFrame with the newest COLOUR frame, injected by the
        #: backend under --record-depth. None = no colour is filed; the depth
        #: recording is unaffected either way.
        self.color_mb = None
        self._fault = ""
        self._grid_fault = ""
        self._grid_fault_said = False
        self._grid_refused_said = False
        self._said_ready = False
        self._t_setup = 0.0
        self.ready_after_s = float("nan")
        # depth ring: dicts {t, depth, obs, stats}
        self._ring: deque = deque(maxlen=self.RING)
        self._depth_marks: deque = deque(maxlen=32)
        self._n_depth = 0
        self._depth_kind = ""
        # proprio
        self._state = None                     # last PolicyState
        self._epoch = None
        self._w_hist: deque = deque(maxlen=512)   # (t_fix, width)
        self._newest_row_fresh = False
        self._first_pending = True             # first inference may duplicate
        self._was_active = False
        # plans
        self._plan_id = 0
        self._last_infer = 0.0
        self._frame_pending = False            # a frame arrived, not yet attempted
        self._plan_marks: deque = deque(maxlen=32)
        self._hz = 0.0
        self._infer_ms = float("nan")
        self._infer_hist: deque = deque(maxlen=1000)
        self._ckpt_sha1 = ""
        self._infer_fault = ""
        self._last_status_pub = 0.0
        self._note = ""
        #: Why the last panel checkpoint pick was REFUSED (PolicyStatus.
        #: ckpt_note), "" after a successful swap. The panel shows it in red;
        #: a refusal that only reached the log was invisible beside a name
        #: that silently snapped back (review 2026-09-11).
        self._ckpt_note = ""
        #: Set by teardown(): set_ckpt refuses afterwards. teardown KEEPS
        #: self.session (a meta() racing it from the controller's thread
        #: still gets a describable object), so "session is None" cannot be
        #: the not-running test.
        self._torn_down = False
        self.counters = {
            "plans": 0, "skip_degenerate": 0, "skip_pair": 0, "skip_fresh": 0,
            "skip_history": 0, "skip_no_depth": 0, "skip_stale_depth": 0,
            "skip_no_start": 0, "skip_grid": 0, "skip_not_ready": 0,
            "skip_fix_lag": 0,
            "pair_dup": 0, "pair_fallback": 0, "pair_near": 0,
            "infer_errors": 0, "epoch_changes": 0, "depth_frames": 0,
            "states": 0, "history_rows": 0,
            # checkpoint swaps made from the panel picker (set_ckpt); 0 means
            # the launch checkpoint (hw_mpc.yaml policy.ckpt / opts) flew.
            "ckpt_swaps": 0,
        }
        self._pair_dts: deque = deque(maxlen=1000)
        # --record-depth: the ONLY way any underwater depth reaches disk. Pure
        # observer, on its own thread, dropping rather than blocking — see
        # perception/depth_record.py. None unless the flag was given.
        self.depth_rec = None
        self._rec_started = False
        #: Set at every PolicyState epoch change (every ENGAGE and every
        #: policy ARM bump it); consumed by _obs_for, which then asks ONCE
        #: whether the controller's run folder moved (_rotate_recorder_if_
        #: moved). Per epoch, not per frame: the check may cost a directory
        #: scan in one race (see there), and a recorder must never pay on
        #: every frame.
        self._rec_check_pending = False
        #: describe() of every recorder closed by a folder rotation, oldest
        #: first — meta()["depth_record"]["previous"], so a run record whose
        #: depth was split across folders SAYS so (review 2026-09-11).
        self._rec_previous: list = []
        #: Why there is no recorder, when the operator asked for one. Empty
        #: means they did not ask. Read by meta() — a run record that says
        #: "--record-depth was not given" when it WAS is the same silent
        #: no-op the flag's refusal exists to prevent.
        self._rec_why = ""

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        self._t_setup = now()
        from ..control.geometry import MpcConfig, NavConfig, S_FLU_FRD
        from ..control.policy_frames import (EtaHistory, T_body_tcp,
                                             tcp_offset_from_body)
        from ..perception.dp_policy import make_policy_session
        from ..perception.policy_obs import DepthObsBuilder

        if self.cfg is None:
            self.cfg = MpcConfig.load(getattr(self.opts, "mpc_config",
                                              "config/hw_mpc.yaml"))
        pc = getattr(self.cfg, "policy", None)
        if pc is None:
            # See _FALLBACK_POLICY_BLOCK: the config class predates the block.
            pc = dict(_FALLBACK_POLICY_BLOCK)
            self.policy_block_source = "backends/policy.py fallback (MpcConfig has no 'policy')"
            self.bus.log.emit("warn", "policy: MpcConfig has no `policy` block — "
                                      "flying on the spec v2 defaults built into "
                                      "backends/policy.py")
        else:
            self.policy_block_source = "MpcConfig.policy"
        # opts.policy_ckpt / --policy-repo override the block (make_policy_session
        # reads them too); mirrored into pc so meta() records what flew. The
        # --policy-ckpt FLAG is gone (2026-09-11, operator request: the
        # checkpoint is chosen in the trajectory panel, set_ckpt below), but
        # tests and tools still set the attribute directly on their Opts
        # class for the stub session, so the read stays getattr-guarded.
        if getattr(self.opts, "policy_ckpt", None):
            pc["ckpt"] = str(self.opts.policy_ckpt)
        if getattr(self.opts, "policy_repo", None):
            pc["repo"] = str(self.opts.policy_repo)
        # Mirrored so meta() records what actually flew, not what the config
        # asked for. Which state dict was loaded is a RECORD BOUNDARY: runs on
        # ema_model and on model are not comparable (see dp_policy's
        # WEIGHT_SOURCES for the held-out numbers).
        if getattr(self.opts, "policy_weights", None):
            pc["weights"] = str(self.opts.policy_weights)
        self.pc = pc

        if self.nav_cfg is None:
            self.nav_cfg = NavConfig.load(
                getattr(self.opts, "nav_config", "config/hw_nav.yaml"),
                geometry_override=getattr(self.opts, "nav_geometry", None))
        R_bc, t_bc = self.nav_cfg.R_t_frd_cam("main")
        self._R_bc = np.asarray(R_bc, float).reshape(3, 3).copy()
        off = pc.get("tcp_offset_cam_m")
        if off is None:
            t_bt = S_FLU_FRD @ np.asarray(pc["tcp_body_flu_m"], float)
            off = tcp_offset_from_body(R_bc, t_bc, t_bt)
        self.tcp_offset_cam_m = np.asarray(off, float).reshape(3)
        self.T_bt = T_body_tcp(R_bc, t_bc, self.tcp_offset_cam_m)
        self.t_body_tcp = self.T_bt[:3, 3].copy()

        self.hist = EtaHistory(keep_s=3.0)
        target = str(pc["target_model"])
        if not Path(target).is_absolute() and not Path(target).exists():
            target = str(REPO / target)
        # TEMPORARY view shift (policy.obs_view_forward_m, default null = off;
        # policy_obs module docstring). The config names a distance along the
        # BODY x axis because that is what a ruler measures on the vehicle; the
        # builder wants it in the camera frame, and R_bc's first row is body x
        # seen from the camera.
        self.obs_view_shift_cam_m = None
        fwd = pc.get("obs_view_forward_m")
        if fwd:
            self.obs_view_shift_cam_m = (float(fwd) * self._R_bc[0, :]).tolist()
        self.builder = DepthObsBuilder(
            target_model=target, z_near=float(pc["z_near_m"]),
            z_far=float(pc["z_far_m"]), out_res=int(pc["obs_res"]),
            warp_size=WARP_SIZE, min_coverage=float(pc["min_obs_coverage"]),
            fill_iters=2, view_shift_cam_m=self.obs_view_shift_cam_m)
        if self.obs_view_shift_cam_m is not None:
            self.bus.log.emit(
                "warn",
                f"policy: OBS VIEW SHIFT {float(fwd):+.3f} m along body x "
                f"(camera frame {np.round(self.obs_view_shift_cam_m, 4).tolist()}) "
                f"— the depth obs is re-rendered from a virtual viewpoint, NOT "
                f"the training recipe (TEMPORARY, policy.obs_view_forward_m; "
                f"null turns it off). Do not pool these runs with unshifted ones.")

        # --record-depth. Constructed here, opened on the first frame
        # (_start_recorder). Nothing else in a run writes depth to disk, so
        # without this flag the run leaves no depth behind at all — not the
        # obs, not the millimetres, and not the pairs it was computed from.
        #
        # runstore.run_dir JOINS the folder the other writers are already using
        # (90 s window), so the recording lands beside that run's CSVs rather
        # than in a folder of its own.
        # The whole block is guarded: an ImportError or a read-only filesystem
        # here must not stop the policy worker from starting. setup() returning
        # early would leave self.enabled False and the mailbox unwanted, i.e. a
        # dead policy for the whole run because a DIAGNOSTIC could not load.
        if getattr(self.opts, "record_depth", False):
            try:
                self.depth_rec = self._new_depth_rec()
            except Exception as e:                               # noqa: BLE001
                self.depth_rec = None
                # The run record must not go on to claim the flag was never
                # given: that is the silent no-op --record-depth exists to end,
                # defeated one layer further in. meta() prints this instead.
                self._rec_why = (f"--record-depth WAS given but the recorder "
                                 f"could not be constructed: "
                                 f"{type(e).__name__}: {e}")
                self.bus.log.emit("warn", f"policy: {self._rec_why}; the run "
                                          f"continues WITHOUT any depth on disk")

        factory = self._session_factory or make_policy_session
        self.session = factory(pc, self.opts)
        self.enabled = True
        self.mailbox.set_wanted(True)
        self.session.start_async(on_log=self._say)
        # The stub is ready synchronously: its (mount-free) contract is
        # checked here; a real checkpoint's is checked when tick() first sees
        # it ready.
        self._check_mount()
        stub = bool(self.session.describe().get("stub", False))
        self.bus.log.emit(
            "info",
            f"policy: {'STUB' if stub else 'loading'} {pc['ckpt']} "
            f"(eval_transforms={pc['eval_transforms']}, steps="
            f"{pc['num_inference_steps']}, period {float(pc['period_s']):.2f} s); "
            f"TCP = ROV jaw at body FRD {np.round(self.t_body_tcp, 3).tolist()} "
            f"(|t| {float(np.linalg.norm(self.t_body_tcp)):.3f} m; handheld "
            f"comparison {list(HANDHELD_TCP_OFFSET_CAM_M)} in ITS camera frame). "
            f"Inference runs only while a policy mission is ACTIVE.")
        self._publish_status(force=True)

    def _new_depth_rec(self):
        """One DepthRecorder on the CONTROLLER's run folder. Called by
        setup() (the launch recorder) and by _rotate_recorder_if_moved()
        (one per later engagement whose folder differs, review 2026-09-11).
        Raises when it cannot be built; the callers decide what that costs.

        The folder is ``run_dir_fn`` = MpcWorker._run_dir, and nothing else,
        for two reasons the controller owns and this worker does not:

         * The TREE. --land-dry-run and LOW level None (--policy-observe)
           write to data/*/*_landdry/ and data/*/*_observe/, and
           MpcWorker._run_tree calls that "a safety property of the RECORD,
           not a tidiness preference". Resolving runstore.DEFAULT_BASE here
           would drop an in-air or hand-dragged recording into the WATER
           tree — inside the 90 s join window of a real pool run, whose
           plans.jsonl is already cited as the provenance for the safety
           limit v_max_m_s.
         * The FOLDER. `_run_dir` is PINNED at engagement, so the recording
           joins the folder the CSVs are already in rather than re-deriving
           one from a clock that has moved on. First inference can be well
           over 90 s after the other writers opened their files.

        A missing ``run_dir_fn`` is therefore a REFUSAL to record, never a
        fallback to runstore.
        """
        from ..perception.depth_record import (DepthRecorder,
                                               DEFAULT_MAX_FRAMES)
        if self.run_dir_fn is None:
            raise RuntimeError(
                "no run_dir_fn was injected — the recorder must file "
                "its frames in the controller's run folder, and this "
                "backend did not provide it")
        return DepthRecorder(
            self.run_dir_fn,
            save_depth=not getattr(self.opts, "record_depth_obs_only", False),
            save_rgb=(self.color_mb is not None
                      and bool(getattr(self.opts, "record_depth_rgb", True))),
            rgb_quality=int(getattr(self.opts, "record_depth_rgb_quality", 85)),
            max_frames=int(getattr(self.opts, "record_depth_max", None)
                           or DEFAULT_MAX_FRAMES),
            on_log=lambda lvl, msg: self.bus.log.emit(lvl, msg))

    def _rotate_recorder_if_moved(self) -> None:
        """Follow the controller's run folder across engagements.

        Review 2026-09-11 (safety + record lenses): the writer thread resolves
        ``run_dir_fn()`` ONCE, at the first frame, and pins the folder
        (frame_record.FrameRecorder._open); ``_start_recorder`` arms once per
        worker life. So every later engagement's frames were appended to the
        FIRST engagement's ``policy_obs/`` with continuing seq numbers. While
        the tree was fixed at launch that was a tidiness defect (two
        engagements > JOIN_WINDOW_S apart). With LOW level None selectable at
        runtime (MpcWorker.set_mode("none")) the two engagements can be in
        DIFFERENT TREES — data vs
        data/*/*_observe — and the tree is the pooling guard
        (_run_tree): an observe run's open-loop frames would be filed inside
        a closed-loop run's folder, under a meta.json naming that run's
        checkpoint, while its own CSV / meta / plans.jsonl went to the other
        tree; render_run_scene then reports "no depth" on the observe run and
        drops the frames as "another engagement's" on the closed-loop run.

        WHEN: once per epoch (every ENGAGE and every policy ARM), from
        _obs_for, on this worker's thread while ``_active()`` holds — the
        controller is engaged, its auto CSV is open, and ``_run_dir()`` is a
        plain attribute read of the pinned folder (workers.py _run_dir): no
        filesystem call. The one race — the controller disengaged between its
        last PolicyState and this inference — makes ``_run_dir`` fall back to
        runstore.run_dir, which JOINS the folder that CSV just closed in: the
        same folder, so nothing rotates, at the cost of one directory scan —
        the same scan the writer thread pays at its first frame — once per
        epoch, never per frame.

        SAME folder -> nothing happens. A fresh recorder would re-open
        index.csv with "w" and clobber the earlier frames, and the same
        folder is the COMMON case: runstore joins within 90 s ("that went
        badly, again"), and a re-START inside one engagement is the same
        pinned folder by construction. A checkpoint swap inside one folder
        is recorded by note_ckpt_swap instead (set_ckpt).

        DIFFERENT folder -> the old recorder is closed (bounded, its meta.json
        written, its describe() kept under meta()["depth_record"]
        ["previous"]) and a new one takes its place; _start_recorder then
        arms it with the CURRENT checkpoint / obs_dt / builder. The
        reference is swapped, never set to None: meta() reads it from the
        controller's thread. close() drops frames still QUEUED at that
        moment (frame_record's bounded-shutdown rule, visible as offered >
        written in the old meta.json); the previous engagement ended
        seconds before this START, so that queue is empty in practice.

        A recorder that never opened (``dir`` None: disabled on an
        unwritable folder, or its first frame still queued) is left alone —
        there is no pinned folder to have moved from, and its ``why`` is the
        run record's honest answer.
        """
        rec = self.depth_rec
        if rec is None or not self._rec_started or self.run_dir_fn is None:
            return
        old_dir = rec.dir
        if old_dir is None:
            return
        try:
            cur = Path(self.run_dir_fn())
        except Exception as e:                                   # noqa: BLE001
            self.bus.log.emit(
                "warn", f"policy: depth recorder could not re-check the run "
                        f"folder ({type(e).__name__}: {e}); frames stay in "
                        f"{old_dir}")
            return
        if cur == Path(old_dir).parent:
            return
        try:
            new = self._new_depth_rec()
        except Exception as e:                                   # noqa: BLE001
            self.bus.log.emit(
                "warn", f"policy: the run folder moved to {cur} but a new "
                        f"depth recorder could not be built "
                        f"({type(e).__name__}: {e}); frames stay in {old_dir}")
            return
        c = rec.close()                    # bounded: CLOSE_TIMEOUT_S ~ 1 s
        self._rec_previous.append(rec.describe())
        self.depth_rec = new
        self._rec_started = False
        self.bus.log.emit(
            "info", f"policy: depth recorder ROTATED — the controller's run "
                    f"folder moved from {Path(old_dir).parent} to {cur} (new "
                    f"engagement); {c['written']} frames stay in {old_dir}, "
                    f"this engagement's frames open {cur / new.SUBDIR}")

    def _note_rec_ready(self) -> None:
        """The session just went READY: meta.json's `extra` follows its
        contract (obs_dt_s) and identity (ckpt_sha1) — after a panel swap
        the values captured at start() were the previous network's."""
        rec = self.depth_rec
        if rec is None or not self._rec_started:
            return
        rec.note_extra("obs_dt_s", self._obs_dt())
        rec.note_extra("ckpt_sha1", self._ckpt_sha1)

    def _say(self, text: str) -> None:
        self.bus.log.emit("info", f"policy: {text}")

    # --------------------------------------------------------------- proprio
    @Slot(object)
    def on_policy_state(self, st) -> None:
        """One PolicyState per controller tick. Rows are appended ONLY when
        ``t_fix`` advances (v2 A6); an epoch change clears everything (A8)."""
        self.counters["states"] += 1
        epoch = int(getattr(st, "epoch", 0))
        if self._epoch is None or epoch != self._epoch:
            if self._epoch is not None:
                self.counters["epoch_changes"] += 1
            self._epoch = epoch
            if self.hist is not None:
                self.hist.clear()
            self._w_hist.clear()
            self._newest_row_fresh = False
            self._first_pending = True
            self._rec_check_pending = True      # see _rotate_recorder_if_moved
        active = bool(st.active) and not bool(st.halted)
        if active and not self._was_active:
            self._first_pending = True          # first inference after START
        self._was_active = active
        self._state = st
        t_fix = getattr(st, "t_fix", None)
        if t_fix is None or self.hist is None:
            return
        if self.hist.append(float(t_fix), st.eta):
            self.counters["history_rows"] += 1
            self._w_hist.append((float(t_fix), float(st.grip_width_m)))
            self._newest_row_fresh = bool(st.fix_fresh)
            cutoff = float(t_fix) - self.hist.keep_s
            while len(self._w_hist) > 2 and self._w_hist[0][0] < cutoff:
                self._w_hist.popleft()

    def _width_at(self, t: float) -> float:
        if not self._w_hist:
            st = self._state
            return float(st.grip_width_m) if st is not None else float("nan")
        ts = np.asarray([w[0] for w in self._w_hist], float)
        ws = np.asarray([w[1] for w in self._w_hist], float)
        return float(np.interp(float(t), ts, ws))

    # ----------------------------------------------------------------- depth
    def _ingest_depth(self) -> bool:
        """Take the mailbox frame into the ring. True when one arrived."""
        item = self.mailbox.take()
        if item is None:
            return False
        self.counters["depth_frames"] += 1
        self._frame_pending = True
        self._n_depth += 1
        t_in = now()
        self._depth_marks.append(t_in)
        grid, kind = self.mailbox.grid()
        b = self.builder
        if b is not None and b.grid_kind == "" and grid is not None:
            from ..perception.policy_obs import GridError
            try:
                b.set_grid(grid)
                self._depth_kind = kind
                self.bus.log.emit(
                    "info", f"policy: depth grid {kind}:{grid.fingerprint} — "
                            f"obs coverage {100.0 * b.coverage:.2f}% "
                            f"(min {100.0 * b.min_coverage:.1f}%)")
            except GridError as e:
                self._grid_fault = str(e)
                self._depth_kind = kind
                if not self._grid_fault_said:
                    self._grid_fault_said = True
                    self.bus.log.emit("error", f"policy: depth grid REFUSED — {e}")
        self._ring.append({"t": float(item["t_capture"]), "depth": item["depth"],
                           "obs": None, "stats": None, "t_arrive": t_in})
        c = self.mailbox.counters()
        if c.get("grid_refused", 0) and not self._grid_refused_said:
            self._grid_refused_said = True
            self.bus.log.emit(
                "error", "policy: a producer offered a DIFFERENT depth grid "
                         "mid-run (calibration/rectification changed) — it is "
                         "not adopted; frames on it are dropped")
        return True

    def _obs_for(self, entry: dict):
        if entry["obs"] is None:
            entry["obs"], entry["stats"] = self.builder.build(entry["depth"])
            # Record on FIRST build only, so each unique depth frame lands
            # once even though the pairing rule asks for it twice. The
            # recorder copies and returns immediately; it cannot raise, block
            # or apply backpressure here (depth_record.DepthRecorder.submit).
            if self.depth_rec is not None:
                if self._rec_check_pending:
                    # Once per epoch, BEFORE arming: did the controller's run
                    # folder move (a new engagement, possibly in the other
                    # tree)? Review 2026-09-11.
                    self._rec_check_pending = False
                    self._rotate_recorder_if_moved()
                self._start_recorder()
                # The colour frame is taken at RECORD time, not at build time:
                # it is the newest the camera has, and its own stamp rides with
                # it so the index can say how far apart the pair was.
                # NEAREST in time to the depth frame, not the newest the
                # camera has: the depth an obs is built from was captured
                # before FoundationStereo ran on it, so "newest" is
                # systematically later (measured +195 ms median, 0907_133358).
                img, t_img = (self.color_mb.nearest(entry["t"])
                              if self.color_mb is not None else (None, 0.0))
                self.depth_rec.submit(entry["t"], entry["obs"], entry["depth"],
                                      color=img, color_t=t_img)
        return entry["obs"], entry["stats"]

    def _start_recorder(self) -> None:
        """Arm the recorder on the first frame, not at setup.

        Deferred for two reasons. The builder's ``describe()`` is only
        meaningful once a grid has been adopted — before that it reports the
        IDLE sentinel, and a meta.json saying ``grid_kind: ""`` would
        misdescribe every frame under it. And the run folder should be named
        when the vehicle is demonstrably flying, not when the station launched.

        ``_rec_started`` needs no lock: the only caller is ``_obs_for`` <-
        ``_infer_once`` <- ``tick``, all on the policy worker's own thread. A
        second caller from any other thread would double-arm and must take a
        lock instead. It is reset only by ``_rotate_recorder_if_moved``
        (same thread), which replaced ``depth_rec`` with a recorder for a
        NEW run folder; the values below are then the current ones.

        ``self.depth_rec`` is never set back to None here. ``meta()`` reads it
        from the controller's thread, and a recorder that reports its own
        failure through ``describe()`` is strictly more informative than a
        None that cannot say whether the feature was off or broken.
        """
        if self._rec_started or self.depth_rec is None:
            return
        self._rec_started = True
        try:
            self.depth_rec.start(
                builder_describe=(self.builder.describe()
                                  if self.builder is not None else {}),
                extra={"ckpt": str(self.pc.get("ckpt", "")),
                       "ckpt_sha1": self._ckpt_sha1,
                       "obs_dt_s": self._obs_dt(),
                       "depth_src": self._depth_kind,
                       "source": str(getattr(self.opts, "source", "hw"))})
        except Exception as e:                                   # noqa: BLE001
            self.bus.log.emit("warn", f"policy: depth recorder failed to "
                                      f"start ({type(e).__name__}: {e}); the "
                                      f"run continues without it")

    def _pick_pair(self, obs_dt: float, allow_dup: bool):
        """(partner entry, how) per v2 A18, or (None, "skip")."""
        newest = self._ring[-1]
        t_d = newest["t"]
        target = t_d - obs_dt
        older = [e for e in list(self._ring)[:-1] if e["t"] < t_d]
        if older:
            best = min(older, key=lambda e: abs(e["t"] - target))
            if abs(best["t"] - target) <= PAIR_TOL * obs_dt:
                return best, "near"
            far = [e for e in older
                   if e["t"] < target - PAIR_TOL * obs_dt
                   and (t_d - e["t"]) <= PAIR_FALLBACK * obs_dt]
            if far:
                return max(far, key=lambda e: e["t"]), "fallback"
        if allow_dup:
            return newest, "dup"
        return None, "skip"

    # ------------------------------------------------------------------ tick
    def _active(self) -> bool:
        st = self._state
        if st is None:
            return False
        if now() - float(st.stamp) > STATE_STALE_S:
            return False
        return bool(st.active) and not bool(st.halted) and bool(st.engaged)

    def _obs_dt(self) -> float:
        s = self.session
        if s is not None and s.ready:
            try:
                return float(s.contract["obs_dt_s"])
            except (KeyError, TypeError):
                pass
        from ..control.geometry import POLICY_DOWN_SAMPLE_STEPS
        return POLICY_DOWN_SAMPLE_STEPS / float(self.pc.get("dataset_fps", 30.0))

    def _action_repr(self) -> str:
        """The checkpoint contract's action representation ("" until the
        session is ready) -- rides on every PolicyPlan and PolicyStatus so
        the controller can refuse a checkpoint whose action it does not fly
        (state.POLICY_ACTION_REPR), the obs_dt pattern."""
        s = self.session
        if s is not None and s.ready:
            try:
                return str(s.contract.get("action_repr", "") or "")
            except (AttributeError, TypeError):
                pass
        return ""

    def _check_mount(self) -> None:
        """The C3 MOUNT check (2026-09-07): a ``pos_yaw_width`` checkpoint
        records the rotation its yaw label was defined on
        (``shape_meta.action.yaw_axis_R_frd_cam`` = the training-side
        ``R_BT_C3``, transcribed from hw_nav.yaml's cam_xyaxes_flu at 43.3
        deg). ``compose_plan`` decodes ``dyaw`` about the body vertical
        THROUGH this side's ``R_bc`` (``T_bt`` above), so if hw_nav's mount
        is re-measured and moves, every dyaw would be flown about a
        different axis than it was labelled on. ``np.allclose(atol=1e-6)``
        against ``R_t_frd_cam('main')``; the result rides on
        ``PolicyStatus.mount_ok`` / ``mount_why`` and the controller refuses
        to ARM on a mismatch (control/workers.py ``_policy_refusal``). No
        yaw-axis keys (legacy checkpoint, stub) = ok: the arm check then
        rests on ``action_repr`` alone -- EXCEPT for ``pos_rpy_width``
        (the 6-DoF variant, 2026-09-26): all three angles are defined
        through ``R_bt``, so the yaw-axis keys are REQUIRED (absent ->
        ``mount_ok`` False, no legacy-permissive path) and the contract's
        ``rpy_convention`` must equal ``dp_policy.RPY_CONVENTION_STATION``
        (the decode in policy_frames is that convention and nothing else).
        A 5-dim / pose10d checkpoint takes exactly the pre-variant path.
        Never raises (called from tick)."""
        s = self.session
        if s is None or not s.ready or self._R_bc is None:
            return
        try:
            c = s.contract
        except (AttributeError, TypeError):
            return
        R_ck = c.get("yaw_axis_R_frd_cam")
        repr_ = str(c.get("action_repr", "") or "")
        if repr_ == ACTION_REPR_POS_RPY_WIDTH:
            why = self._rpy_contract_why(c, R_ck)
            if why:
                self._mount_ok, self._mount_why = False, why
                if not self._mount_said:
                    self._mount_said = True
                    self.bus.log.emit("error", f"policy: MOUNT MISMATCH -- {why}")
                return
        if R_ck is None:
            self._mount_ok, self._mount_why = True, ""
            return
        try:
            R_ck = np.asarray(R_ck, float).reshape(3, 3)
        except (ValueError, TypeError):
            self._mount_ok = False
            self._mount_why = (f"checkpoint yaw_axis_R_frd_cam is malformed "
                               f"({c.get('yaw_axis_R_frd_cam')!r})")
        else:
            if np.allclose(R_ck, self._R_bc, atol=1e-6):
                self._mount_ok, self._mount_why = True, ""
            else:
                self._mount_ok = False
                self._mount_why = (
                    f"checkpoint yaw axis R_frd_cam (cam_tilt "
                    f"{c.get('yaw_axis_cam_tilt_deg')} deg) != hw_nav "
                    f"R_frd_cam('main'), max |dR| "
                    f"{float(np.abs(R_ck - self._R_bc).max()):.2e} > 1e-6 -- "
                    f"the dyaw label was defined on the TRAINING mount; "
                    f"retrain on the measured mount or restore hw_nav.yaml "
                    f"cam_xyaxes_flu; refusing to arm")
        if not self._mount_ok and not self._mount_said:
            self._mount_said = True
            self.bus.log.emit("error", f"policy: MOUNT MISMATCH -- {self._mount_why}")

    @staticmethod
    def _rpy_contract_why(c: dict, R_ck) -> str:
        """Why a ``pos_rpy_width`` contract may NOT arm ("" = it may, as far
        as the keys go; the R comparison itself follows in _check_mount).
        Two rules, both without a permissive path: the yaw-axis keys must be
        present (roll and pitch are labelled through the same ``R_bt`` as
        yaw), and ``rpy_convention`` must equal the station's constant
        (``order`` / ``frame`` strings, ``columns`` in the same order)."""
        if R_ck is None or c.get("yaw_axis_cam_tilt_deg") is None:
            return (f"{ACTION_REPR_POS_RPY_WIDTH} checkpoint declares no "
                    f"yaw_axis_R_frd_cam / yaw_axis_cam_tilt_deg -- roll, pitch "
                    f"and yaw are all defined through that mount rotation, so "
                    f"an undeclared mount cannot be flown; refusing to arm")
        conv = c.get("rpy_convention")
        if not isinstance(conv, dict):
            return (f"{ACTION_REPR_POS_RPY_WIDTH} checkpoint declares no "
                    f"shape_meta.action.rpy_convention (want "
                    f"{RPY_CONVENTION_STATION}); refusing to arm")
        want = {k: (list(v) if isinstance(v, (list, tuple)) else str(v))
                for k, v in RPY_CONVENTION_STATION.items()}
        got = {k: (list(v) if isinstance(v, (list, tuple)) else str(v))
               for k, v in conv.items()}
        if got != want:
            return (f"{ACTION_REPR_POS_RPY_WIDTH} checkpoint rpy_convention "
                    f"{got} != the station's {want} -- the decode "
                    f"(policy_frames.decode_pos_rpy) is that convention only; "
                    f"refusing to arm")
        return ""

    # ------------------------------------------- checkpoint swap (the picker)
    @staticmethod
    def _same_ckpt(a, b) -> bool:
        """Two checkpoint spellings name the same session: a stub spelling
        (``stub`` / ``stub7`` / ``stub_rp``, dp_policy.STUB_CKPT_VARIANTS)
        only equals the SAME spelling; two existing files compare RESOLVED
        (a symlinked ``selected.ckpt`` and its target are one network)."""
        a, b = str(a or "").strip(), str(b or "").strip()
        if not a or not b:
            return False
        if is_stub_ckpt(a) or is_stub_ckpt(b):
            return a.lower() == b.lower()
        pa, pb = Path(a).expanduser(), Path(b).expanduser()
        try:
            if pa.exists() and pb.exists():
                return pa.resolve() == pb.resolve()
        except OSError:
            pass
        return a == b

    def _refuse_ckpt(self, why: str) -> None:
        """One refusal: the log line AND the status (PolicyStatus.ckpt_note)
        — the panel shows the note in red; the log alone was invisible next
        to a name that silently snapped back (review 2026-09-11)."""
        self._ckpt_note = f"checkpoint swap REFUSED — {why}"
        self.bus.log.emit("warn", f"policy: {self._ckpt_note}")
        self._publish_status(force=True, note="ckpt refused")

    @Slot(str)
    def set_ckpt(self, path) -> None:
        """Load a different checkpoint, chosen in the trajectory panel
        (``bus.cmd_policy_ckpt``; 2026-09-11, operator request: the GUI
        picker replaces the removed ``--policy-ckpt`` flag).

        Runs on the worker thread (a queued TimerWorker slot), i.e. the same
        thread as ``tick``, so a swap never races an inference. The
        preconditions, in this order (each refusal is logged at warn and
        published with ``ckpt_note`` set; ``_refuse_ckpt``):

        1. the worker is not running (stopping / torn down / setup never
           finished);
        2. an empty path is ignored silently;
        3. the same path as the held session: nothing to do, status
           republished (no note);
        4. the CONSUMER's state: a stale PolicyState means a silent
           controller (wait); ``engaged`` means a mission may be armed —
           DISENG first. No state at all (no controller, or not ready yet)
           allows the swap: nothing can be armed;
        5. the held session is still loading (``MODEL_LOAD_LOCK`` serialises
           loads; a second loader would double the GPU memory);
        6. not ``stub`` and not a file;
        7. FoundationStereo parity, the SAME rule as ``__main__.check_policy``
           (only under ``--fstereo``): a training store that disagrees with
           the flown FS settings is a WARNING listing the mismatches (the
           shipped --policy defaults are themselves off the store, so a
           refusal here would refuse every real pick under the default
           launch); an unreadable store is a warning too.

        Then the swap: the old session is closed (frees torch, empties the
        CUDA cache), the per-session state is reset, ``pc["ckpt"]`` follows
        the pick (meta / the depth recorder read it), the new session starts
        loading asynchronously and the status says ``loading`` until the tick
        sees it READY (or ERROR). ``counters["ckpt_swaps"]`` counts it and
        ``meta()["ckpt_source"]`` becomes ``panel``.
        """
        path = str(path or "").strip()
        # 1. not running
        if (self.stopping or self._torn_down or not self.enabled
                or self.builder is None):
            self._refuse_ckpt("the policy worker is not running")
            return
        # 2. empty
        if not path:
            self.bus.log.emit("debug", "policy: empty checkpoint pick ignored")
            return
        s = self.session
        name = Path(path).name if not is_stub_ckpt(path) else path.lower()
        # 3. same session
        if s is not None and self._same_ckpt(getattr(s, "ckpt", ""), path):
            self._ckpt_note = ""
            self.bus.log.emit("info", f"policy: {name} is already the loaded "
                                      f"checkpoint")
            self._publish_status(force=True)
            return
        # 4. the consumer's state
        st = self._state
        if st is not None:
            if now() - float(st.stamp) > STATE_STALE_S:
                self._refuse_ckpt("the controller is silent (its last "
                                  "PolicyState is stale) — wait for it")
                return
            if bool(st.engaged):
                self._refuse_ckpt("DISENG first (a mission may be armed; the "
                                  "checkpoint is pinned at ARM)")
                return
        # 5. still loading
        if s is not None and getattr(s, "loading", False):
            held = getattr(s, "ckpt", "")
            held = Path(held).name if not is_stub_ckpt(held) else str(held).lower()
            self._refuse_ckpt(f"still loading {held}; wait for READY or ERROR")
            return
        # 6. a file
        if not is_stub_ckpt(path) and not Path(path).expanduser().is_file():
            self._refuse_ckpt(f"not a file: {path}")
            return
        # The swap is going ahead: no mission can be running (4.), so leave
        # the schedule gate open BEFORE the slow steps below — the parity read
        # and the old session's close() run inside this slot, with no tick in
        # between to do it.
        self._fs_idle()
        # 7. FS parity (check_policy's rule, at pick time)
        if bool(getattr(self.opts, "fstereo", False)):
            tds = training_depth_source(path)
            if tds.get("status") == "ok":
                bad = fs_settings_mismatch(self.opts, tds)
                if bad:
                    self.bus.log.emit(
                        "warn", f"policy: {name}: flying with FS settings the "
                                f"policy did NOT train on — " + "; ".join(bad)
                                + " (recorded in the meta; "
                                  "--policy-allow-fs-mismatch "
                                  f"{'given' if getattr(self.opts, 'policy_allow_fs_mismatch', False) else 'not given'})")
            elif tds.get("status") != "stub":
                self.bus.log.emit(
                    "warn", f"policy: {name}: FS-vs-training parity NOT checked "
                            f"({tds.get('status')}"
                            f"{': ' + tds['error'] if tds.get('error') else ''})")
        # the swap
        from ..perception.dp_policy import make_policy_session

        old = self.session
        prev_ckpt = str(self.pc.get("ckpt", ""))
        self.session = None
        if old is not None:
            try:
                old.close()
            except Exception as e:                               # noqa: BLE001
                self.bus.log.emit("warn", f"policy: closing the old session "
                                          f"raised {type(e).__name__}: {e}")
        self._said_ready = False
        self.ready_after_s = float("nan")
        self._t_setup = now()
        self._ckpt_sha1 = ""
        self._infer_fault = ""
        self._fault = ""
        self._mount_ok, self._mount_why, self._mount_said = True, "", False
        self._first_pending = True
        self._last_infer = 0.0
        self._frame_pending = False
        self.pc["ckpt"] = path
        factory = self._session_factory or make_policy_session
        try:
            self.session = factory(self.pc, self.opts, ckpt=path)
        except Exception as e:                                   # noqa: BLE001
            # The old session is already closed: the worker now holds NO
            # network. Say so through the status (error) rather than raise
            # out of a slot; the next pick starts from `session is None`.
            self._fault = (f"checkpoint swap failed to build a session for "
                           f"{name}: {type(e).__name__}: {e}")
            self._ckpt_note = self._fault
            self.bus.log.emit("error", f"policy: {self._fault}")
            self._publish_status(force=True, note="ckpt failed")
            return
        self.session.start_async(on_log=self._say)
        # The stub is ready synchronously (contract checked here); a real
        # checkpoint's mount is checked when tick() first sees it ready.
        self._check_mount()
        self.counters["ckpt_swaps"] += 1
        self._ckpt_note = ""
        # --record-depth: the recorder stays armed across the swap (same run
        # folder, runstore joins within 90 s), and its meta.json captured
        # `extra.ckpt` ONCE at the first frame — so it would name the
        # previous network for every frame this one consumes (review
        # 2026-09-11). Note the swap at the seq boundary instead; no file is
        # touched here (meta.json is written at close).
        rec = self.depth_rec
        if rec is not None and self._rec_started:
            entry = rec.note_ckpt_swap(prev_ckpt, path,
                                       swap_no=self.counters["ckpt_swaps"])
            if entry is not None:
                self.bus.log.emit(
                    "info", f"policy: depth recorder: frames from seq "
                            f"{entry['next_seq']} on belong to {name} "
                            f"(policy_obs/meta.json extra.ckpt_swaps)")
        stub = bool(self.session.describe().get("stub", False))
        self.bus.log.emit(
            "info", f"policy: {'STUB' if stub else 'loading'} {name} (chosen in "
                    f"the panel; swap #{self.counters['ckpt_swaps']}, the "
                    f"previous session is closed)")
        self._publish_status(
            force=True,
            note=("loading" if self.session.loading
                  else ("ready" if self.session.ready else "not ready")))

    def tick(self) -> None:
        s = self.session
        if s is None or not self.enabled:
            self._fs_idle()
            self._publish_status()
            return
        self._ingest_depth()
        if s.error:
            self._fs_idle()
            self._publish_status(note=s.error[:60])
            return
        if not s.ready:
            self._fs_idle()
            self._publish_status(note="loading" if s.loading else "not ready")
            return
        if not self._said_ready:
            self._said_ready = True
            self.ready_after_s = now() - self._t_setup
            d = s.describe()
            self._ckpt_sha1 = str(d.get("ckpt_sha1_head", "") or "")
            self._check_mount()
            self._note_rec_ready()
            # The session logs its own load/warm-up numbers (they are
            # assigned a moment AFTER `ready` flips, so reading them here
            # would print nan); this line is the worker's clock.
            self.bus.log.emit(
                "info", f"policy: READY after {self.ready_after_s:.1f} s "
                        f"(obs_dt {self._obs_dt() * 1e3:.1f} ms, fps source: "
                        f"{s.contract.get('fps_source', '?')}; action "
                        f"{self._action_repr() or '?'}; mount "
                        f"{'ok' if self._mount_ok else 'MISMATCH'}; sha1 "
                        f"{self._ckpt_sha1 or '-'})")
        if not self._active():
            self._fs_idle()
            self._first_pending = True
            self._publish_status(note="idle")
            return
        t = now()
        period = float(self.pc.get("period_s", 0.5))
        gate = self.fs_gate
        if gate is not None and gate.mode == "only":
            if self._burst is not None or not self._only_blocked():
                # The heartbeat `only` keys on: while these calls keep coming
                # FoundationStereo computes only what this worker asks for;
                # when they stop, it runs free again.
                gate.set_active(True)
                self._tick_only(gate, t, period)
                return
            # No forward could run whatever frames arrived (no fresh fix, no
            # proprio history yet, no grid): a burst would be computed and
            # thrown away. Let FoundationStereo run free and fall through to
            # the `free` trigger, which counts the skip on each arrival
            # exactly as a `free` run does — it cannot fire a forward here,
            # because _skip_reason() refuses on the same conditions.
            gate.set_active(False)
        if t - self._last_infer < period:
            self._publish_status()
            return
        # Armed: an attempt needs a frame that ARRIVED since the last one. A
        # frame that arrived while the period was still running leaves the
        # flag set, so in practice this fires on the first tick after the
        # period, on the newest frame already in the ring; it waits for an
        # arrival only when none came since the last attempt. A skip does not
        # stamp _last_infer: the next frame retries.
        if not self._frame_pending:
            self._publish_status()
            return
        self._frame_pending = False
        skip = self._skip_reason()
        if skip:
            self.counters[skip] += 1
            self._publish_status(note=skip)
            return
        if gate is not None and gate.mode == "only":
            # Not reachable while _only_blocked() and _skip_reason() agree
            # (the blocked branch above is the only way here under `only`).
            # If they ever drift apart, do not run a forward beside a
            # free-running FoundationStereo and call it `only`: the next tick
            # takes the burst path.
            self._publish_status()
            return
        if self._infer_once(t):
            self._last_infer = t

    def _fs_idle(self) -> None:
        """No policy mission is running here: tell the gate (under `only`
        FoundationStereo then runs free) and forget any burst and hold. Called
        on EVERY path that leaves the mission — not active, loading, a failed
        session, a checkpoint swap, teardown — because the gate's own
        heartbeat timeout is the backstop, not the mechanism."""
        gate = self.fs_gate
        if gate is None:
            return
        self._burst = None
        gate.set_active(False)
        gate.release()

    def _tick_only(self, gate, t: float, period: float) -> None:
        """One tick of an active mission under `--policy-fs-schedule only`.

        FoundationStereo is idle between bursts, so nothing arrives unless
        this asks: ``ONLY`` the request makes frames. The request goes out
        ``_only_lead_s`` before the period ends; the attempt fires when
        BURST_FRAMES frames have ARRIVED since the request (the pair
        ``_pick_pair`` then takes is those two) AND the period has elapsed —
        period_s is the lower bound on the interval between attempts in every
        schedule. The ``free`` trigger's arrival flag is not consulted — a
        frame left over from before the request would otherwise fire an
        attempt on a stale observation.

        Every way out without a forward (a skip, a burst that did not
        complete) leaves ``_last_infer`` unstamped and ``_burst`` None, so the
        next tick asks again: FoundationStereo is never left waiting for a
        request that will not come. (tick() does not come here at all while
        ``_only_blocked()`` — FoundationStereo then runs free.)
        """
        b = self._burst
        if b is None:
            # Never a lead longer than the period itself: with a short
            # period_s that would ask again the moment an attempt ended and
            # the plans would come faster than the config says.
            lead = min(self._only_lead_s, max(0.0, period - ONLY_LEAD_MIN_S))
            if t - self._last_infer < period - lead:
                self._publish_status()
                return
            if gate.request_burst(BURST_FRAMES):
                self._burst = {"t": t}
                self._frame_pending = False
                self._sched_n["bursts"] += 1
            else:
                self._sched_n["burst_refused"] += 1
            self._publish_status()
            return
        t_req = float(b["t"])
        # `>`: a frame ingested on the request's own tick was taken BEFORE
        # the request (tick() ingests, then stamps t), so it is not counted.
        got = sum(1 for e in self._ring
                  if float(e.get("t_arrive", 0.0) or 0.0) > t_req)
        if got < BURST_FRAMES:
            if t - t_req > float(gate.burst_timeout_s):
                # The frames did not come (no pair from the camera, or this
                # thread was blocked while two arrived and the one-slot
                # mailbox kept only the newer). Ask again — and say so: this
                # is a plan that is late by more than a period.
                self._burst = None
                self._sched_n["burst_timeouts"] += 1
                if t - self._burst_warn_t > 5.0:
                    self._burst_warn_t = t
                    self.bus.log.emit(
                        "warn", f"policy: FoundationStereo delivered {got} of "
                                f"{BURST_FRAMES} frames within "
                                f"{float(gate.burst_timeout_s):.1f} s of the "
                                f"request (--policy-fs-schedule only) — asking "
                                f"again ({self._sched_n['burst_timeouts']} so "
                                f"far)")
                self._publish_status(note="burst_timeout")
                return
            self._publish_status()
            return
        if "t_done" not in b:
            # The burst is in. Track how long it took, for the next lead —
            # down at once, up slowly: a lead longer than the burst makes the
            # frames wait for the period below (an older observation), while
            # a lead that is too short only makes the plan a little late.
            b["t_done"] = t
            dur = t - t_req
            lead = (dur if dur < self._only_lead_s
                    else 0.8 * self._only_lead_s + 0.2 * dur)
            self._only_lead_s = min(ONLY_LEAD_MAX_S, max(ONLY_LEAD_MIN_S, lead))
        if t - self._last_infer < period:
            # period_s stays the LOWER bound on the interval between
            # attempts, as it is under `free` and `yield`: a burst that came
            # in early (the lead was longer than this burst took) waits here.
            self._publish_status()
            return
        self._burst = None
        self._frame_pending = False
        skip = self._skip_reason()
        if skip:
            self.counters[skip] += 1
            self._publish_status(note=skip)
            return
        if self._infer_once(t):
            self._last_infer = t

    def _only_blocked(self) -> bool:
        """`only`: would an attempt be skipped for a reason NO depth frame
        can cure — no usable grid, no proprio history, the newest row not a
        fresh fix, no start pose? These are _skip_reason()'s own tests minus
        the two about the depth ring, which under `only` is stale between
        bursts by design."""
        b = self.builder
        if b is None or not b.usable:
            return True
        if self.hist is None or len(self.hist) == 0:
            return True
        if self.hist.span() < self._obs_dt():
            return True
        if not self._newest_row_fresh:
            return True
        st = self._state
        return st is None or st.eta_start is None

    def _fs_input_tag(self) -> str:
        """FoundationStereo's network input as asked for on the command line,
        for the plan record: "size WxH itN" (an explicit size governs) else
        "scale S itN"; "" when this run's depth is not FoundationStereo.
        Alpha, the checkpoint and what the session actually ran are in the
        run meta (fstereo.*), not here."""
        if self.fs_gate is None and self.fstereo_meta_fn is None:
            return ""
        iters = getattr(self.opts, "fstereo_iters", None)
        try:
            it = f" it{int(iters)}" if iters is not None else ""
        except (TypeError, ValueError):
            it = ""
        size = getattr(self.opts, "fstereo_size", None)
        if size:
            return f"size {str(size).lower()}{it}"   # argparse hands "WxH"
        scale = getattr(self.opts, "fstereo_scale", None)
        try:
            return f"scale {float(scale):g}{it}" if scale is not None else ""
        except (TypeError, ValueError):
            return ""

    def _skip_reason(self) -> str:
        """The counter name of the first A9/A6/A18 gate that fails, or ''."""
        b = self.builder
        if b is None or not b.usable:
            return "skip_grid"
        if not self._ring:
            return "skip_no_depth"
        t_d = self._ring[-1]["t"]
        if now() - t_d > float(self.pc.get("obs_max_age_s", 0.6)):
            return "skip_stale_depth"
        if self.hist is None or len(self.hist) == 0:
            return "skip_history"
        if self.hist.span() < self._obs_dt():
            return "skip_history"
        if not self._newest_row_fresh:
            return "skip_fresh"
        st = self._state
        if st is None or st.eta_start is None:
            return "skip_no_start"
        return ""

    def _infer_once(self, t_trigger=None) -> bool:
        """One attempt. True when the network RAN (a plan, or a predict
        error — both consume the period); False on a pre-forward skip.
        ``t_trigger`` is the tick that began the attempt (the plan record)."""
        from ..control.policy_frames import lowdim_obs

        obs_dt = self._obs_dt()
        newest = self._ring[-1]
        t_d = newest["t"]
        partner, how = self._pick_pair(obs_dt, allow_dup=self._first_pending)
        if partner is None:
            self.counters["skip_pair"] += 1
            self._publish_status(note="skip_pair")
            return False
        # The proprio rows are spaced like the DEPTH pair: a fallback pair
        # 138 ms apart with rows 66.7 ms apart would show the network 2x the
        # image motion of the proprio motion, a combination training never
        # produced (verify 2026-09-02). A duplicated first frame has no image
        # baseline, so its rows keep the training stride.
        pair_dt = float(t_d - partner["t"])
        spacing = pair_dt if how != "dup" else obs_dt
        eta_prev, eta_now, info = _rows_for(self.hist, t_d, spacing)
        if float(info["fix_lag_s"]) > FIX_LAG_TOL * obs_dt:
            self.counters["skip_fix_lag"] += 1
            self._publish_status(note="skip_fix_lag")
            return False
        if info["degenerate"]:
            self.counters["skip_degenerate"] += 1
            self._publish_status(note="skip_degenerate")
            return False
        t_prev, t_now = float(info["t_prev"]), float(info["t_now"])
        st = self._state
        # FS SCHEDULE (2026-10-01; perception/fs_gate.py). Past the three
        # pre-forward skips above the network WILL be asked, so this is where
        # FoundationStereo is told not to start another frame: the observation
        # build below is CPU work that the frame in flight finishes under, and
        # the wait just before the forward covers whatever is left of it. The
        # finally releases on every way out — the two `return True`s, an
        # exception from the unguarded lines between them — because a tick
        # exception is swallowed by TimerWorker._tick and nobody else would;
        # the gate's own lease is the backstop, not the plan. `fs_gate` None
        # (the default) makes no call at all.
        gate = self.fs_gate
        fs_wait_ms = None
        fs_timed_out = None
        if gate is not None:
            gate.hold()
        try:
            try:
                obs_prev, _ = self._obs_for(partner)
                obs_now, stats_now = self._obs_for(newest)
            except Exception as e:                               # noqa: BLE001
                self._infer_error(f"obs build: {type(e).__name__}: {e}")
                return True
            img = np.stack([obs_prev, obs_now], axis=0).astype(np.float32) / 255.0
            img = np.ascontiguousarray(img.transpose(0, 3, 1, 2))     # (2,3,R,R)
            w_prev = self._width_at(t_prev)
            w_now = self._width_at(t_now)
            lowdim = lowdim_obs(eta_prev, eta_now, st.eta_start, self.T_bt,
                                w_prev, w_now)
            obs = {IMAGE_KEY: img}
            obs.update(lowdim)
            if gate is not None:
                # NOT inside infer_ms (the session times its own forward):
                # the wait is its own number on the plan.
                fs_wait_ms, fs_timed_out = gate.wait_idle()
            try:
                action, infer_ms = self.session.predict(obs)
            except Exception as e:                               # noqa: BLE001
                self._infer_error(f"predict: {type(e).__name__}: {e}")
                return True
        finally:
            if gate is not None:
                gate.release()
        self._infer_fault = ""
        self._plan_id += 1
        self.counters["plans"] += 1
        self.counters[f"pair_{how}"] += 1
        self._pair_dts.append(pair_dt)
        self._infer_ms = float(infer_ms)
        self._infer_hist.append(float(infer_ms))
        t_emit = now()
        self._plan_marks.append(t_emit)
        if len(self._plan_marks) >= 2:
            span = self._plan_marks[-1] - self._plan_marks[0]
            self._hz = (len(self._plan_marks) - 1) / span if span > 0 else 0.0
        self._first_pending = False
        plan = PolicyPlan(
            plan_id=self._plan_id, epoch=int(self._epoch or 0),
            obs_t=float(t_d), t_emit=t_emit, infer_ms=float(infer_ms),
            action=np.ascontiguousarray(action, dtype=np.float32),
            lowdim={k: np.array(v, copy=True) for k, v in lowdim.items()},
            obs_rows_t=(t_prev, t_now),
            obs_fix_t=tuple(self.hist.fix_stamps(t_prev))
            + tuple(self.hist.fix_stamps(t_now)),
            obs_pair_dt_s=float(pair_dt), pair_dup=(how == "dup"),
            depth_src=self.builder.grid_kind,
            depth_coverage=float(self.builder.coverage),
            depth_valid=float(stats_now["obs_valid"]),
            ckpt_sha1=self._ckpt_sha1, obs_dt_s=float(obs_dt),
            action_repr=self._action_repr(),
            depth_arrive_t=newest.get("t_arrive"),
            trigger_t=(float(t_trigger) if t_trigger is not None else None),
            fs_schedule=(gate.mode if gate is not None else "free"),
            fs_wait_ms=(float(fs_wait_ms) if fs_wait_ms is not None else None),
            fs_wait_timeout=(bool(fs_timed_out) if fs_timed_out is not None
                             else None),
            fs_input=self._fs_input_tag(), stamp=t_emit)
        self.bus.policy_plan.emit(plan)
        self._publish_status(force=True, note=f"plan {self._plan_id} ({how})")
        return True

    def _infer_error(self, text: str) -> None:
        self.counters["infer_errors"] += 1
        if text != self._infer_fault:
            self._infer_fault = text
            self.bus.log.emit("warn", f"policy: {text}")
        self._publish_status(note=text[:60])

    # ---------------------------------------------------------------- status
    def _conn(self) -> Conn:
        s = self.session
        if s is None or self._fault:
            return Conn.FAULT
        if s.error:
            return Conn.FAULT
        if s.loading or not s.ready:
            return Conn.CONNECTING
        if self._grid_fault or self._infer_fault:
            return Conn.DEGRADED
        if not self._ring or now() - self._ring[-1]["t"] > 2.0:
            return Conn.DEGRADED
        return Conn.ONLINE

    def _depth_hz(self) -> float:
        if len(self._depth_marks) < 2:
            return 0.0
        span = self._depth_marks[-1] - self._depth_marks[0]
        return (len(self._depth_marks) - 1) / span if span > 0 else 0.0

    def _publish_status(self, note: str = "", force: bool = False) -> None:
        """PolicyStatus + the SENSORS row, >= 1 Hz when idle (v2 A13)."""
        t = now()
        if note:
            self._note = note          # kept even when this publish is rate-limited
        if not force and t - self._last_status_pub < 1.0:
            return
        self._last_status_pub = t
        s = self.session
        n_skip = sum(v for k, v in self.counters.items() if k.startswith("skip_"))
        st = PolicyStatus(
            ready=bool(s is not None and s.ready),
            loading=bool(s is not None and s.loading),
            error=(s.error if s is not None and s.error else self._fault),
            hz=float(self._hz), infer_ms=float(self._infer_ms),
            depth_src=self.builder.grid_kind if self.builder is not None else "",
            note=self._note, n_plans=int(self.counters["plans"]),
            n_skip=int(n_skip),
            # The grid's own state, apart from the session's: a REFUSED grid
            # leaves depth_src set and ready True, and the controller's
            # refusal must see it (verify 2026-09-02).
            grid_ok=bool(self.builder is not None and self.builder.usable),
            grid_why=(self.builder.why if self.builder is not None else "no builder"),
            obs_dt_s=(self._obs_dt() if s is not None and s.ready else 0.0),
            action_repr=self._action_repr(),
            mount_ok=bool(self._mount_ok), mount_why=str(self._mount_why),
            # WHICH checkpoint the worker holds (the panel picker's truth and
            # the controller's ARM pin, 2026-09-11): the session's own path
            # while one exists, else the block's (a swap whose session could
            # not be built still names what was asked for).
            ckpt=(str(getattr(s, "ckpt", "") or "") if s is not None
                  else str(self.pc.get("ckpt", "") or "")),
            ckpt_sha1=str(self._ckpt_sha1),
            ckpt_note=str(self._ckpt_note),
            fs_schedule=(self.fs_gate.mode if self.fs_gate is not None
                         else "free"),
            conn=self._conn(), stamp=t)
        self.bus.policy_status.emit(st)
        detail = self._sensor_detail(st)
        self.bus.sensor_stat.emit(SensorStat(
            "DP policy", (self._hz if self._hz > 0 else None), st.conn, detail))

    def _sensor_detail(self, st: PolicyStatus) -> str:
        if st.error:
            return st.error[:28]
        if st.loading:
            return "loading"
        if not st.ready:
            return "not ready"
        d = f"{st.depth_src or 'no depth'} {self._depth_hz():.0f}fps"
        if self._grid_fault:
            return f"{d} grid REFUSED"
        if st.n_plans:
            return f"{d} {st.infer_ms:.0f}ms #{st.n_plans}"
        if not self._note:
            return d
        # Canonical short notes ride whole ('idle', a bare skip_* counter
        # name); anything longer is cut so the row never wraps.
        note = self._note
        if note != "idle" and not note.startswith("skip_"):
            note = note[:14]
        return f"{d} {note}"

    # ------------------------------------------------------------------ meta
    def _rec_describe(self, rec) -> dict:
        """``rec.describe()`` plus, when a folder rotation closed earlier
        recorders this worker life, ``previous`` (their describe(), oldest
        first) and ``rotations`` — so a run record whose depth is split across
        folders says so instead of pointing at one folder (review 2026-09-11).
        ``self._rec_previous`` is appended on the policy thread and copied
        here on the controller's; a list copy under the GIL cannot tear."""
        d = rec.describe()
        prev = list(self._rec_previous)
        if prev:
            d["previous"] = prev
            d["rotations"] = len(prev)
        return d

    def meta(self) -> dict:
        """The policy block of the run record (``mpc.policy_meta_fn``).
        Plain fields only — called from the controller's thread."""
        s = self.session
        b = self.builder
        _rec = self.depth_rec          # snapshot; see "depth_record" below
        pc = dict(self.pc)
        stub = bool(s is not None and s.describe().get("stub", False))
        src = str(getattr(self.opts, "source", "hw"))
        pair = np.asarray(self._pair_dts, float) if self._pair_dts else np.zeros(0)
        infer = np.asarray(self._infer_hist, float) if self._infer_hist else np.zeros(0)
        fs = None
        if self.fstereo_meta_fn is not None:
            try:
                fs_all = self.fstereo_meta_fn() or {}
                fs = {k: fs_all.get(k) for k in
                      ("iters", "scale", "infer_size", "infer_size_actual",
                       "ckpt", "ckpt_sha1_first_8mib", "measured_hz")}
                rig = fs_all.get("rig") or {}
                fs["alpha"] = rig.get("alpha") if isinstance(rig, dict) else None
            except Exception as e:                               # noqa: BLE001
                fs = {"error": f"{type(e).__name__}: {e}"}
        out = {
            "enabled": True,
            # A LAND DRY-RUN counts as synthetic: the network and the depth
            # are real, but the localizer is a fixed pose and the vehicle
            # cannot move, so nothing in this record measures the policy's
            # closed-loop behaviour. Readers already treat `synthetic` as
            # "do not cite", which is exactly the right verdict here.
            "synthetic": bool(stub or src == "demo"
                              or getattr(self.opts, "land_dry_run", False)),
            "land_dry_run": bool(getattr(self.opts, "land_dry_run", False)),
            "source": src,
            "policy_block_source": self.policy_block_source,
            "session": s.describe() if s is not None else {"error": self._fault},
            "obs": b.describe() if b is not None else None,
            "depth_src": self._depth_kind,
            "depth_scale_applied": (float(getattr(self.opts, "depth_scale",
                                                  DEVICE_DEPTH_SCALE_DEFAULT)
                                          or DEVICE_DEPTH_SCALE_DEFAULT)
                                    if self._depth_kind == "color_aligned" else None),
            "grid_fault": self._grid_fault,
            # Whether any depth left this run on disk, and how much of it. A
            # reader asking "can I look at that run's depth" must be able to
            # answer from the meta alone; `enabled: false` means the answer is
            # no and no amount of digging will change it.
            # Snapshotted into a local first: this runs on the CONTROLLER's
            # thread while the policy thread may be in teardown, and a
            # `self.depth_rec` that changed between the test and the reads
            # would raise AttributeError — which workers.py catches by
            # replacing this ENTIRE policy meta block with an error string,
            # losing the session, obs recipe and TCP geometry for exactly the
            # run that misbehaved.
            "depth_record": (self._rec_describe(_rec) if _rec is not None else
                             {"enabled": bool(self._rec_why),
                              "why": self._rec_why or
                                     "--record-depth was not given; this run "
                                     "persisted no depth map and no stereo "
                                     "pair, so its depth cannot be inspected "
                                     "or regenerated"}),
            "obs_dt_s": self._obs_dt(),
            "period_s": float(pc.get("period_s", float("nan"))),
            "tcp": {
                "tcp_body_flu_m": pc.get("tcp_body_flu_m"),
                "tcp_offset_cam_m": (self.tcp_offset_cam_m.tolist()
                                     if self.tcp_offset_cam_m is not None else None),
                "tcp_offset_cam_m_explicit": pc.get("tcp_offset_cam_m") is not None,
                "handheld_tcp_offset_cam_m": list(HANDHELD_TCP_OFFSET_CAM_M),
                "t_body_tcp_frd_m": (self.t_body_tcp.tolist()
                                     if self.t_body_tcp is not None else None),
                "t_body_tcp_norm_m": (float(np.linalg.norm(self.t_body_tcp))
                                      if self.t_body_tcp is not None else None),
                "T_body_tcp": self.T_bt.tolist() if self.T_bt is not None else None,
            },
            "fstereo": {
                "requested": {"iters": getattr(self.opts, "fstereo_iters", None),
                              "scale": getattr(self.opts, "fstereo_scale", None),
                              "size": getattr(self.opts, "fstereo_size", None),
                              "alpha": getattr(self.opts, "fstereo_alpha", None)},
                "worker": fs,
                "training": training_depth_source(pc.get("ckpt", "")),
                "allow_mismatch": bool(getattr(self.opts, "policy_allow_fs_mismatch",
                                               False)),
            },
            "ready_after_s": self.ready_after_s,
            # Where the flown checkpoint came from (2026-09-11 panel picker):
            # "launch" = hw_mpc.yaml policy.ckpt / opts, "panel" = set_ckpt
            # swapped it at least once. `session.ckpt` above is the path.
            "ckpt_swaps": int(self.counters["ckpt_swaps"]),
            "ckpt_source": "panel" if self.counters["ckpt_swaps"] else "launch",
            "counters": dict(self.counters),
            "pairing": {
                "n": int(pair.size),
                "dt_p50_s": float(np.percentile(pair, 50)) if pair.size else None,
                "dt_max_s": float(pair.max()) if pair.size else None,
                "dup": int(self.counters["pair_dup"]),
                "fallback": int(self.counters["pair_fallback"]),
                "near": int(self.counters["pair_near"]),
                "skip": int(self.counters["skip_pair"]),
                "tol": PAIR_TOL, "fallback_max": PAIR_FALLBACK,
                "skip_fix_lag": int(self.counters["skip_fix_lag"]),
                "fix_lag_tol": FIX_LAG_TOL,
                "trigger": ("depth-frame arrival after period_s"
                            if getattr(self.fs_gate, "mode", "free") != "only"
                            else f"{BURST_FRAMES} frames requested from "
                                 f"FoundationStereo ahead of period_s "
                                 f"(--policy-fs-schedule only)"),
                # `trigger` above is kept word for word for `free`/`yield`
                # (every record since 2026-09-02 carries it), but it
                # overstates: what the code does is this. Per plan,
                # trigger_age_s - depth_ready_age_s in plans.jsonl is how old
                # the newest frame already was when the attempt began.
                "trigger_detail": (
                    "first 20 ms tick after period_s, on the newest frame "
                    "already in the ring; waits for an arrival only when none "
                    "came since the last attempt"
                    if getattr(self.fs_gate, "mode", "free") != "only" else
                    "when the requested frames have arrived and period_s has "
                    "elapsed; free trigger while no forward could run"),
            },
            # Which FoundationStereo schedule this run flew (2026-10-01).
            # RECORD BOUNDARY: plan age and infer_ms of runs on different
            # schedules must not be pooled, and under `only` the pair spacing
            # (pairing.near/fallback above, obs_pair_dt_s per plan) and the
            # trigger differ too. Always written, `free` included — an absent
            # block is a build that had no switch.
            "fs_schedule": self._fs_schedule_meta(),
            "infer_ms": {
                "p50": float(np.percentile(infer, 50)) if infer.size else None,
                "max": float(infer.max()) if infer.size else None,
                "n": int(infer.size),
            },
            "depth_hz": self._depth_hz(),
            "mailbox": self.mailbox.counters(),
            "plan_hz": float(self._hz),
            "state_stale_s": STATE_STALE_S,
        }
        return out

    def _fs_schedule_meta(self) -> dict:
        """What was typed, what is in effect, why they differ, and the gate's
        own counters. Runs on the CONTROLLER's thread (``meta``): the gate's
        snapshot copies under its own short lock, and nothing here may raise
        (an exception replaces the whole policy block with an error line)."""
        gate = self.fs_gate
        requested = str(getattr(self.opts, "policy_fs_schedule", "free") or "free")
        effective = str(getattr(gate, "mode", "free")) if gate is not None else "free"
        out = {
            "requested": requested, "effective": effective,
            "why": ("" if requested == effective else
                    "no FoundationStereo worker feeds this policy (demo source "
                    "or device depth): there is nothing to schedule, so the "
                    "run is `free`" if gate is None else
                    f"a gate was injected with mode {effective}; the options "
                    f"said {requested}"),
            "worker": dict(self._sched_n),
            "only_lead_s": (round(float(self._only_lead_s), 3)
                            if effective == "only" else None),
            "gate": None,
        }
        if gate is not None:
            try:
                out["gate"] = gate.snapshot()
            except Exception as e:                               # noqa: BLE001
                out["gate"] = {"error": f"{type(e).__name__}: {e}"}
        return out

    # -------------------------------------------------------------- teardown
    def teardown(self) -> None:
        # set_ckpt refuses from here on; self.session is KEPT (closed, not
        # dropped) for a meta() racing this from the controller's thread.
        self._torn_down = True
        # This worker stops BEFORE the stereo worker (Backend.stop is reverse
        # order): leave the gate open behind it.
        self._fs_idle()
        self.mailbox.set_wanted(False)
        self.enabled = False
        if self.depth_rec is not None:
            # close() is idempotent and bounded (CLOSE_TIMEOUT_S ~ 1 s, well
            # inside the 4 s worker-stop budget). The reference is KEPT so a
            # meta() racing this teardown from the controller's thread still
            # gets a describable object rather than a None.
            try:
                self.depth_rec.close()
            except Exception:                                    # noqa: BLE001
                pass
        if self.session is not None:
            try:
                self.session.close()
            except Exception:                                    # noqa: BLE001
                pass


__all__ = ["PolicyWorker", "rect_left_grid_from_rig", "training_depth_source",
           "fs_settings_mismatch", "effective_fstereo_ckpt", "hydra_config_for",
           "PAIR_TOL", "PAIR_FALLBACK", "FIX_LAG_TOL", "STATE_STALE_S",
           "IMAGE_KEY"]
