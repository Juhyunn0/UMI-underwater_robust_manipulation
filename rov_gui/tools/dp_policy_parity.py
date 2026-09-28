#!/usr/bin/env python3
"""dp_policy_parity.py — is the station's policy THE trained policy?

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tools/dp_policy_parity.py \
        [--ref rov_gui/tests/data/dp_policy_ref.npz] [--ckpt ...] [--repo ...]

The station side of a two-interpreter comparison. ``dp_policy_reference.py``
(run in ``umi2``, the training environment, through the UPSTREAM policy
class) pushed one fixed synthetic observation through the checkpoint and
saved every intermediate that a re-implementation can get wrong:

    nobs_*      the normalizer outputs per key           -> exact (fp32 ULPs)
    tokens      the obs encoder's (1, 106, 768) tokens   -> max |diff| <= 5e-3 (*)
    eps0        model(traj0, timesteps[0], tokens)       -> max |diff| <= 1e-3
    step0       one DDIM step from that fixed state      -> max |diff| <= 1e-3
    nsample /   the full seeded 16-step sample and the   -> REPORTED, not
    action      unnormalized action                         load-bearing
    action_dim / action_repr  the reference's contract    -> must equal the
                (fixtures written before 2026-09-07 lack    station's (skipped
                them: skipped with a note)                  with a note if absent)

``traj0`` / ``cond_data`` / ``noise_probe`` are ``(1, 16, D)`` with ``D`` the
checkpoint's action width (10 pose10d, 5 pos_yaw_width, 7 pos_rpy_width --
the 6-DoF variant, 2026-09-26, fixture ``dp_policy_ref_pos_rpy_width.npz``
written by ``dp_policy_reference.py --out`` on that checkpoint; the 5-dim
fixture is untouched); the station's probe takes ``D`` from its own
contract, never from the file.

This script loads the same checkpoint through ``rov_gui.perception.dp_policy``
(``rovgui-pose``: torch 2.11 / timm 1.0.29 / diffusers 0.40 versus the
reference's 2.9 / 0.9.7 / 0.18) and compares. The deterministic rows are the
proof: the encoder ran with an Identity transform on both sides (the
reference replaced RandomCrop/RandomRotation by Identity; here
``eval_transforms="identity"``), and eps0 / step0 are evaluated from the
REFERENCE's tokens and traj0 so each stage is checked in isolation (the
encoder, the denoiser, the scheduler) as well as end-to-end.

(*) Why the token tolerance is 5e-3 and not 1e-3: the reference tokens were
computed with cuDNN's default TF32 convolutions (10-bit mantissa), and so
are ours, on a different cuDNN. The same weights evaluated in float64 on the
CPU differ from the reference by 3.6e-3 and from the station's GPU tokens by
5.1e-3 [측정 2026-09-02, this script's "tokens vs fp64 CPU" row on the RTX
5090]; the station-vs-reference gap of 2.4e-3 is INSIDE that noise floor, so
a 1e-3 gate would fail two correct implementations of the same network. The
gate that carries the proof is therefore the next row: the denoiser fed with
OUR tokens must match the reference eps0 within 1e-3 (measured 5e-5) — the
token noise is immaterial downstream — and the token row's fp64 comparison is
printed every run so the floor is never assumed.

The seeded full sample is reported and not asserted: whether two torch
versions draw the same ``randn`` stream from the same seed is a fact about
torch, not about this loader — ``noise_probe`` in the reference is the first
draw of that stream, so the table says WHICH of the two happened (stream
identical -> the samples must agree; stream different -> they cannot, and
the deterministic rows carry the proof alone).

The last row loads the checkpoint the way flight does (``center``) and runs
``predict`` twice with the same seed: identical output proves dropout and
BatchNorm are in eval mode on the thread that predicts.

Exit status 1 on any load-bearing mismatch.

Regenerating the fixture for a new checkpoint (the 5-dim retrain)
-----------------------------------------------------------------
The fixture directory ``rov_gui/tests/data/`` is gitignored (``.gitignore:34
data/``; verify with ``git check-ignore -v rov_gui/tests/data/dp_policy_ref.npz``),
so the old file is preserved by a plain ``mv`` — nothing to commit::

    mv rov_gui/tests/data/dp_policy_ref.npz \
       rov_gui/tests/data/dp_policy_ref_pose10d_20260902.npz
    ~/miniforge3/envs/umi2/bin/python rov_gui/tools/dp_policy_reference.py \
        --ckpt <new run>/checkpoints/selected.ckpt \
        --out rov_gui/tests/data/dp_policy_ref.npz
    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tools/dp_policy_parity.py \
        --ckpt <new run>/checkpoints/selected.ckpt \
        | tee rov_gui/tools/dp_policy_out/parity_<date>.log

Both ``--ckpt`` default to ``rov_gui.state.POLICY_CKPT_DEFAULT`` once that
points at the new run; ``--ref`` names the file to compare against, so the
preserved pose10d fixture still runs against the old checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.perception import dp_policy                       # noqa: E402
from rov_gui.state import (ACTION_REPR_BY_DIM, ACTION_REPR_POS_RPY_WIDTH,  # noqa: E402
                           ACTION_REPR_POS_YAW_WIDTH, ACTION_REPR_POSE10D)
from rov_gui.tools.dp_policy_reference import (safe_eval as ref_safe_eval,  # noqa: E402
                                               synthetic_obs)

#: Column units per representation (the parity table's "action" row label):
#: pos(3) m, then the angle column(s) rad, then the width m; rot6d for pose10d.
ACTION_UNITS = {ACTION_REPR_POS_YAW_WIDTH: "m/rad/m",
                ACTION_REPR_POS_RPY_WIDTH: "m/rad/rad/m",
                ACTION_REPR_POSE10D: "m/rot6d/m"}

TOL = 1e-3          # deterministic tensors (eps0, step0), max |diff|
TOL_TOKENS = 5e-3   # encoder tokens: TF32 conv noise floor, see docstring (*)
TOL_NORM = 1e-5     # normalizer: the same fp32 multiply-add, ULP noise only


def _row(name, ok, detail, load_bearing=True, tag=None):
    if tag is None:
        tag = ("ok  " if ok else "FAIL") if load_bearing else ("agree" if ok else "differ")
    return f"  {tag:6s} {name:34s} {detail}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=str(Path(__file__).resolve().parents[1]
                                        / "tests" / "data" / "dp_policy_ref.npz"))
    ap.add_argument("--ckpt", default=dp_policy.DEFAULT_CKPT)
    ap.add_argument("--repo", default=dp_policy.DEFAULT_UMI_REPO)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=None,
                    help="state dict to load on the station side; default = the "
                         "shipped policy.weights (2026-09-07: model)")
    a = ap.parse_args()

    ref_path = Path(a.ref)
    if not ref_path.is_file():
        print(f"reference file missing: {ref_path} — run "
              f"rov_gui/tools/dp_policy_reference.py in umi2 first")
        return 2
    ref = np.load(ref_path)
    seed = int(ref["seed"])
    steps = int(ref["steps"])
    print(f"reference: {ref_path}")
    print(f"  written with torch {ref['torch_version']} from {ref['ckpt']}")
    print(f"  ckpt sha1 head {str(ref['ckpt_sha1_head'])[:12]}, seed {seed}, "
          f"steps {steps}, timesteps {ref['timesteps'].tolist()}")

    # the two safe_eval copies must agree (one in the tool, one in the module)
    for expr in ("2*3/4", "-(1+2)*3", "16 * 2 - 1"):
        assert dp_policy.safe_eval(expr) == ref_safe_eval(expr), expr

    import torch
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable — the parity comparison needs the GPU "
              "(the reference was computed on one); skipping (exit 0)")
        return 0

    rows = []
    failed = False

    # ----------------------------------------------------------------- load
    from rov_gui.control.geometry import default_policy_block
    weights = a.weights or str(default_policy_block().get("weights", "ema_model"))
    s = dp_policy.DpPolicySession(a.ckpt, a.repo, num_inference_steps=steps,
                                  eval_transforms="identity", device=a.device,
                                  weights=weights)
    s.load(lambda m: print("  " + m))
    d = s.describe()
    # A fixture built on the OTHER state dict disagrees everywhere downstream of
    # the load, so check it before reporting a wall of red.
    r_w = str(ref["weights"]) if "weights" in ref.files else ""
    rows.append(_row("weights", (not r_w) or r_w == weights,
                     f"station {weights!r} vs ref {r_w!r}" if r_w else
                     f"station {weights!r}; ref fixture predates the key — regenerate",
                     load_bearing=bool(r_w), tag=None if r_w else "skip"))
    failed |= bool(r_w) and r_w != weights
    same_ckpt = d["ckpt_sha1_head"] == str(ref["ckpt_sha1_head"])
    rows.append(_row("checkpoint sha1 head", same_ckpt,
                     f"{d['ckpt_sha1_head'][:12]} vs ref "
                     f"{str(ref['ckpt_sha1_head'])[:12]}"))
    failed |= not same_ckpt
    same_ts = d["timesteps"] == ref["timesteps"].tolist()
    rows.append(_row("DDIM timesteps", same_ts, f"{d['timesteps']}"))
    failed |= not same_ts

    # the action contract: the station's (from the checkpoint it loaded) vs
    # the reference's (from the checkpoint the fixture was written for)
    c = s.contract
    c_dim = int(c["action_dim"])
    c_repr = str(c.get("action_repr") or ACTION_REPR_BY_DIM.get(c_dim, ""))
    units = ACTION_UNITS.get(c_repr, "m/rot6d/m")
    if "action_dim" in ref.files:
        r_dim = int(ref["action_dim"])
        r_repr = str(ref["action_repr"]) if "action_repr" in ref.files else ""
        r_eff = r_repr or ACTION_REPR_BY_DIM.get(r_dim, "")
        same_c = (c_dim == r_dim) and (c_repr == r_eff)
        rows.append(_row("action contract", same_c,
                         f"station dim {c_dim} {c_repr!r} vs ref dim {r_dim} "
                         f"{(r_repr or '(no key: inferred ' + repr(r_eff) + ')')}"))
        failed |= not same_c
    else:
        rows.append(_row("action contract", True,
                         f"ref fixture predates action_dim/action_repr (written "
                         f"before 2026-09-07) — station dim {c_dim} {c_repr!r} "
                         f"unchecked; regenerate per the docstring",
                         load_bearing=False, tag="skip"))

    obs = synthetic_obs(seed)
    for k in obs:
        assert np.array_equal(obs[k], ref[f"obs_{k}"]), f"synthetic obs {k} drifted"

    dev = torch.device(a.device)
    obs_t = {k: torch.as_tensor(v)[None].to(dev) for k, v in obs.items()}
    with torch.no_grad():
        # 1. normalizer, per key
        nobs = s._norm_t.normalize_dict(obs_t)
        for k in obs:
            got = nobs[k].cpu().numpy()
            want = ref[f"nobs_{k}"]
            diff = float(np.abs(got - want).max())
            ok = diff <= TOL_NORM
            rows.append(_row(f"normalize {k}", ok, f"max|diff| {diff:.2e}"))
            failed |= not ok

        # 2. encoder tokens (Identity transform both sides)
        tokens = s._encoder(nobs)
        tok_np = tokens.cpu().numpy()
        diff = float(np.abs(tok_np - ref["tokens"]).max())
        ok = diff <= TOL_TOKENS
        rows.append(_row("encoder tokens", ok,
                         f"{tuple(tokens.shape)} max|diff| {diff:.2e} "
                         f"(tol {TOL_TOKENS}, TF32 floor; see (*))"))
        failed |= not ok
        # the noise floor, measured: the same weights in float64 on the CPU
        import copy
        enc64 = copy.deepcopy(s._encoder).cpu().double()
        nobs64 = {k: v.cpu().double() for k, v in nobs.items()}
        tok64 = enc64(nobs64).numpy()
        d_ref = float(np.abs(tok64 - ref["tokens"]).max())
        d_own = float(np.abs(tok64 - tok_np).max())
        rows.append(_row("tokens vs fp64 CPU (noise floor)", True,
                         f"ref-fp64 {d_ref:.2e}, station-fp64 {d_own:.2e}",
                         load_bearing=False))
        del enc64

        # 3. one denoiser evaluation from the REFERENCE tokens and traj0
        tokens_ref = torch.as_tensor(ref["tokens"]).to(dev)
        traj0 = torch.as_tensor(ref["traj0"]).to(dev)
        t0 = torch.tensor([int(ref["timesteps"][0])], device=dev)
        eps0 = s._model(traj0, t0, tokens_ref)
        diff = float(np.abs(eps0.cpu().numpy() - ref["eps0"]).max())
        ok = diff <= TOL
        rows.append(_row("denoiser eps0 (ref tokens)", ok,
                         f"max|diff| {diff:.2e} (tol {TOL})"))
        failed |= not ok
        eps0_own = s._model(traj0, t0, tokens)
        diff = float(np.abs(eps0_own.cpu().numpy() - ref["eps0"]).max())
        rows.append(_row("denoiser eps0 (own tokens)", diff <= TOL,
                         f"max|diff| {diff:.2e} (tol {TOL})"))
        failed |= not (diff <= TOL)

        # 4. one DDIM step from the reference eps0 (scheduler in isolation)
        eps0_ref = torch.as_tensor(ref["eps0"]).to(dev)
        s._scheduler.set_timesteps(steps)
        step0 = s._scheduler.step(eps0_ref, int(ref["timesteps"][0]),
                                  traj0).prev_sample
        diff = float(np.abs(step0.cpu().numpy() - ref["step0"]).max())
        ok = diff <= TOL
        rows.append(_row("DDIM step0 (ref eps0)", ok,
                         f"max|diff| {diff:.2e} (tol {TOL})"))
        failed |= not ok

        # 5. does this torch draw the reference's randn stream? (probe width
        # from the CONTRACT: the file's width is the reference's, not ours)
        probe = torch.randn((1, int(c["action_horizon"]), c_dim), device=dev,
                            generator=torch.Generator(device=dev).manual_seed(seed))
        probe_np = probe.cpu().numpy()
        if probe_np.shape == ref["noise_probe"].shape:
            stream_same = bool(np.allclose(probe_np, ref["noise_probe"], atol=1e-6))
            rows.append(_row("randn stream (seed) vs reference", stream_same,
                             "identical" if stream_same else
                             "DIFFERENT (torch version) — full sample cannot agree",
                             load_bearing=False))
        else:
            stream_same = False
            rows.append(_row("randn stream (seed) vs reference", False,
                             f"probe {probe_np.shape} vs ref "
                             f"{ref['noise_probe'].shape}: fixture written for "
                             f"another action width — not comparable",
                             load_bearing=False, tag="skip"))

    # 6. the full seeded sample through predict()
    action, ms = s.predict(obs, seed=seed)
    if action.shape != ref["action"][0].shape:
        rows.append(_row("full seeded sample (action)", False,
                         f"station {action.shape} vs ref {ref['action'][0].shape}: "
                         f"fixture written for another action width",
                         load_bearing=False, tag="skip"))
        agree = False
        s.close()
        rows.append(_row("action contract (fixture)", False,
                         "the fixture is not for this checkpoint — regenerate "
                         "per the docstring"))
        failed = True
        print("\n".join(rows))
        print("\nRESULT: MISMATCH — the reference fixture is for another contract")
        return 1
    diff_a = float(np.abs(action - ref["action"][0]).max())
    agree = diff_a <= TOL
    rows.append(_row("full seeded sample (action)", agree,
                     f"max|diff| {diff_a:.2e} {units}, {ms:.1f} ms"
                     + ("" if stream_same else " [stream differs; expected]"),
                     load_bearing=False))
    nsample_own = s._norm_t.normalize("action", torch.as_tensor(action).to(dev))
    diff_n = float(np.abs(nsample_own.cpu().numpy() - ref["nsample"][0]).max())
    rows.append(_row("full seeded sample (nsample)", diff_n <= TOL,
                     f"max|diff| {diff_n:.2e}", load_bearing=False))
    if stream_same and not agree:
        # same noise, same weights, same schedule: this WOULD be a bug
        rows.append(_row("seeded sample with identical stream", False,
                         "streams identical but samples differ"))
        failed = True
    s.close()

    # 7. flight configuration: two predicts, one seed, identical
    s2 = dp_policy.DpPolicySession(a.ckpt, a.repo, num_inference_steps=steps,
                                   eval_transforms="center", device=a.device)
    s2.load()
    a1, ms1 = s2.predict(obs, seed=seed)
    a2, ms2 = s2.predict(obs, seed=seed)
    ident = bool(np.array_equal(a1, a2))
    rows.append(_row("center: two predicts identical", ident,
                     f"max|diff| {float(np.abs(a1 - a2).max()):.2e}, "
                     f"{ms1:.1f} / {ms2:.1f} ms"))
    failed |= not ident
    a3, _ = s2.predict(obs)
    rows.append(_row("center: unseeded predict differs", True,
                     f"max|diff| vs seeded {float(np.abs(a1 - a3).max()):.2e} "
                     f"(diffusion noise; informational)", load_bearing=False))
    s2.close()

    print(f"\nstation: torch {d['torch']}, diffusers {d['diffusers']}, "
          f"timm {d['timm']}, device {d['device']}")
    print("\n".join(rows))
    print("\nRESULT:", "MISMATCH — the station policy is NOT the trained policy"
          if failed else "PARITY — deterministic stages agree within tolerance")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
