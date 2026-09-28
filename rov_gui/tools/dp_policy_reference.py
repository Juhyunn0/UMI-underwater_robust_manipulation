#!/usr/bin/env python3
"""dp_policy_reference.py — REFERENCE outputs of the trained diffusion policy,
computed with the UPSTREAM class in the TRAINING environment.

    ~/miniforge3/envs/umi2/bin/python rov_gui/tools/dp_policy_reference.py \
        --ckpt <run>/checkpoints/selected.ckpt --out rov_gui/tests/data/dp_policy_ref.npz

(run from the repo root; ``--ckpt`` / ``--repo`` default to
``rov_gui.state.POLICY_CKPT_DEFAULT`` / ``POLICY_UMI_REPO``, the ONE checkpoint
path, imported by appending the repo root to ``sys.path`` — ``rov_gui.state``
is stdlib-only so the training env can import it.)

Why this exists
---------------
The station runs the policy in `rovgui-pose` (torch 2.11, timm 1.0.29,
diffusers 0.40) through its own loader (rov_gui/perception/dp_policy.py),
which reimplements the upstream LinearNormalizer and DDIM loop because the
upstream `normalizer.py` cannot be imported there. The checkpoint was trained
in `umi2` (torch 2.9, timm 0.9.7, diffusers 0.18). Two interpreters, two code
paths, one checkpoint: the only honest proof that the station's policy IS the
trained policy is a fixed input pushed through both and compared. This script
is the training-side half; `rov_gui/tools/dp_policy_parity.py` is the station
side and reads the file this writes.

What is fixed, and why
----------------------
* The obs is a seeded synthetic pattern (no dataset needed on either side).
* The encoder's RandomCrop / RandomRotation are replaced by Identity for the
  DETERMINISTIC part of the comparison (tokens, one denoiser evaluation) —
  random crops would differ per process regardless of correctness.
* The full DDIM sample uses a seeded CUDA generator. Whether the two torch
  versions draw the same randn stream is recorded, not assumed: the station
  side compares its own sample with `generator` seeded the same way and reports
  whether they agree; the deterministic tensors are the load-bearing check.
* No autocast, fp32 throughout.
* The action width ``D`` is READ from the checkpoint
  (``cfg.shape_meta.action.shape[0]``: 10 for the legacy pose10d run, 5 for
  the 2026-09-07 pos_yaw_width retrain, 7 for a pos_rpy_width run of the
  6-DoF variant, 2026-09-26) and drives every fixed tensor —
  ``traj0`` / ``cond_data`` / ``noise_probe`` are ``(1, 16, D)``. ``action_dim``
  and ``action_repr`` (``shape_meta.action.action_repr``, "" when the run
  predates the key) are saved in the npz so the station side can refuse a
  fixture written for another contract instead of crashing on a shape; a
  7-dim run also saves its ``rpy_convention`` (JSON string, "" when absent).

A 7-dim fixture goes to its OWN file so the 5-dim one stays untouched:

    ~/miniforge3/envs/umi2/bin/python rov_gui/tools/dp_policy_reference.py \
        --ckpt <run_7d>/checkpoints/selected.ckpt \
        --out rov_gui/tests/data/dp_policy_ref_pos_rpy_width.npz
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import sys
from pathlib import Path

import numpy as np

# Appended LAST so nothing in this checkout can shadow the training env's own
# packages (rov_gui.state is stdlib-only; see the docstring).
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
from rov_gui.state import POLICY_CKPT_DEFAULT, POLICY_UMI_REPO   # noqa: E402

DEFAULT_UMI = POLICY_UMI_REPO
DEFAULT_CKPT = POLICY_CKPT_DEFAULT

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def safe_eval(expr) -> float:
    """Arithmetic-only stand-in for the `${eval:...}` resolver the training
    config uses (train.py registers Python's eval; a checkpoint's config is
    trusted but there is no reason to hand it the interpreter)."""
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


def synthetic_obs(seed: int = 0) -> dict:
    """The fixed input. Shapes follow the checkpoint's shape_meta (no batch
    dim): depth (2,3,224,224) in [0,1] with ch1 a 0/1 validity mask and
    ch2 == ch0 (the dataset's channel convention), lowdim rows (T=2, D)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) / 223.0
    frames = []
    for k in range(2):
        v = 0.55 - 0.35 * yy + 0.10 * np.sin(6.0 * xx + 0.3 * k) \
            + rng.normal(0.0, 0.01, (224, 224)).astype(np.float32)
        v = np.clip(v, 0.0, 1.0).astype(np.float32)
        valid = (rng.random((224, 224)) > 0.03).astype(np.float32)
        v = v * valid
        # match the u8 quantisation the dataset stores
        u8 = np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8).astype(np.float32) / 255.0
        frames.append(np.stack([u8, valid, u8], axis=0))
    depth = np.stack(frames, axis=0).astype(np.float32)          # (2,3,224,224)
    eye6 = np.array([1, 0, 0, 0, 1, 0], np.float32)
    lowdim = {
        "robot0_eef_pos": np.array([[0.004, -0.003, 0.006], [0, 0, 0]], np.float32),
        "robot0_eef_rot_axis_angle": np.stack([
            np.array([0.9998, 0.0100, -0.0150, -0.0098, 0.9999, 0.0120], np.float32),
            eye6]),
        "robot0_gripper_width": np.array([[0.081], [0.080]], np.float32),
        "robot0_eef_rot_axis_angle_wrt_start": np.stack([
            np.array([0.96, 0.20, -0.19, -0.21, 0.95, 0.23], np.float32),
            np.array([0.95, 0.22, -0.22, -0.23, 0.94, 0.25], np.float32)]),
    }
    return {"camera0_depth": depth, **lowdim}


def sha1_head(path: Path, nbytes: int = 8 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()


def _deployed_weights() -> str:
    """The state dict the station flies (policy.weights of the shipped config)."""
    from rov_gui.control.geometry import default_policy_block
    return str(default_policy_block().get("weights", "ema_model"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--repo", default=DEFAULT_UMI)
    ap.add_argument("--out", default="rov_gui/tests/data/dp_policy_ref.npz")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--weights", default=None,
                    help="which state dict to build the fixture on; default = the\n"
                         "DEPLOYED choice (geometry.default_policy_block()['weights'], "
                         "2026-09-07: model). A fixture on weights the station does not "
                         "fly proves nothing about the flown network.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import dill
    import hydra
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", safe_eval, replace=True)

    ckpt = Path(a.ckpt).resolve()
    payload = torch.load(open(ckpt, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    act_meta = cfg.shape_meta.action
    action_dim = int(act_meta.shape[0])
    action_horizon = int(act_meta.get("horizon", 16))
    action_repr = str(act_meta.get("action_repr", "") or "")
    rpy_conv = act_meta.get("rpy_convention", None)
    rpy_convention = (json.dumps(OmegaConf.to_container(rpy_conv, resolve=True))
                      if rpy_conv is not None else "")
    policy = hydra.utils.instantiate(cfg.policy)
    # The DEPLOYED state dict, not a fixed one: selected.json / the policy config
    # decide (2026-09-07 the deployed choice moved from ema_model to model, and a
    # fixture built on the other copy would prove parity of weights nobody flies).
    weights = a.weights or _deployed_weights()
    if weights not in payload["state_dicts"]:
        raise SystemExit(f"checkpoint has no state_dicts[{weights!r}]; it holds "
                         f"{sorted(payload['state_dicts'])}")
    print(f"weights: {weights}")
    missing, unexpected = policy.load_state_dict(payload["state_dicts"][weights],
                                                 strict=False)
    if unexpected or any(not k.startswith("normalizer.") for k in missing):
        raise SystemExit(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    policy.eval().cuda()
    policy.num_inference_steps = int(a.steps)

    obs = synthetic_obs(a.seed)
    dev = torch.device("cuda")
    obs_t = {k: torch.as_tensor(v)[None].to(dev) for k, v in obs.items()}

    # Deterministic part: identity transform, fp32.
    for key in list(policy.obs_encoder.key_transform_map.keys()):
        policy.obs_encoder.key_transform_map[key] = nn.Identity()
    with torch.no_grad():
        nobs = policy.normalizer.normalize(obs_t)
        tokens = policy.obs_encoder(nobs)                                 # (1,N,768)
        scheduler = policy.noise_scheduler
        scheduler.set_timesteps(policy.num_inference_steps)
        timesteps = [int(t) for t in scheduler.timesteps]
        traj0 = torch.linspace(-0.5, 0.5, action_horizon * action_dim,
                               device=dev).reshape(1, action_horizon, action_dim)
        eps0 = policy.model(traj0, torch.tensor([timesteps[0]], device=dev), tokens)
        # One scheduler step from that fixed state (deterministic for DDIM eta=0).
        step0 = scheduler.step(eps0, timesteps[0], traj0).prev_sample
        # Full sample with a seeded generator, then unnormalize.
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        cond_data = torch.zeros((1, action_horizon, action_dim), device=dev)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = policy.conditional_sample(cond_data, cond_mask, cond=tokens,
                                            generator=gen)
        action = policy.normalizer["action"].unnormalize(nsample)
        noise_probe = torch.randn((1, action_horizon, action_dim), device=dev,
                                  generator=torch.Generator(device=dev).manual_seed(a.seed))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        seed=a.seed, steps=a.steps, timesteps=np.asarray(timesteps),
        ckpt=str(ckpt), ckpt_sha1_head=sha1_head(ckpt),
        torch_version=torch.__version__,
        action_dim=action_dim, action_repr=action_repr, weights=weights,
        rpy_convention=rpy_convention,
        **{f"obs_{k}": v for k, v in obs.items()},
        **{f"nobs_{k}": v.cpu().numpy() for k, v in nobs.items()},
        tokens=tokens.cpu().numpy(),
        traj0=traj0.cpu().numpy(), eps0=eps0.cpu().numpy(), step0=step0.cpu().numpy(),
        nsample=nsample.cpu().numpy(), action=action.cpu().numpy(),
        noise_probe=noise_probe.cpu().numpy(),
    )
    print(f"wrote {out}: tokens {tuple(tokens.shape)}, action {tuple(action.shape)} "
          f"[action_dim {action_dim}, action_repr {action_repr!r}"
          f"{', rpy_convention ' + rpy_convention if rpy_convention else ''}], "
          f"torch {torch.__version__}, ckpt sha1[:12] {sha1_head(ckpt)[:12]}")
    print("action[0]:", np.round(action[0, 0].cpu().numpy(), 4))
    print("action[15]:", np.round(action[0, 15].cpu().numpy(), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
