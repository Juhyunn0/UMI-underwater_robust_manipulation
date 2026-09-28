#!/usr/bin/env python3
"""publish_ckpt.py — put a training run's chosen checkpoint into data/checkpoints/.

    python tools/publish_ckpt.py <run_dir> --epoch 195 --why "held-out 219-window replay, model weights"
    python tools/publish_ckpt.py <run_dir> --ckpt <run_dir>/checkpoints/selected.ckpt --why "..."
    python tools/publish_ckpt.py --list

``data/checkpoints/`` is the ONE flat folder holding the best checkpoint of
every training run (operator request 2026-09-14). While a run trains, the
workspace keeps the entry pointing at the top-1 by train_loss (``how: auto``).
This tool is the HUMAN step: after a held-out replay
(``rov_gui/tools/dp_policy_offline.py replay --val-only --weights model``)
picks an epoch, run it here and the entry becomes ``how: selected: <why>`` —
which the workspace never overwrites again. ``--why`` is required for a
selection because the sidecar is the only place the reason survives.

The entry is a relative symlink into the run's own ``checkpoints/`` (nothing
is copied; the run keeps every topk file as before) plus a ``.json`` sidecar.
Also refreshes ``<run>/checkpoints/selected.ckpt`` so the two spellings agree.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BEST_DIR = REPO / "data" / "checkpoints"
UMI = REPO / "external" / "UMI_aquatic"


def _best_ckpt():
    if str(UMI) not in sys.path:
        sys.path.insert(0, str(UMI))
    from diffusion_policy.common import best_ckpt      # noqa: E402
    return best_ckpt


def list_entries() -> int:
    if not BEST_DIR.is_dir():
        print(f"(no {BEST_DIR.relative_to(REPO)} yet)")
        return 0
    for link in sorted(BEST_DIR.glob("*.ckpt")):
        side = {}
        try:
            side = json.loads(link.with_suffix(".json").read_text())
        except (OSError, ValueError):
            pass
        target = os.readlink(link) if link.is_symlink() else "(file)"
        ok = "" if link.exists() else "   !! dangling"
        print(f"{link.name}\n    -> {target}{ok}\n    epoch {side.get('epoch')}  "
              f"{side.get('how', '?')}  ({side.get('published_at', '?')})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", nargs="?", help="the training run folder (data/<day>/<...>_train_<...>)")
    ap.add_argument("--epoch", type=int, help="epoch of the checkpoint in <run>/checkpoints/")
    ap.add_argument("--ckpt", help="explicit checkpoint file (instead of --epoch)")
    ap.add_argument("--why", help="how it was selected (required unless --auto)")
    ap.add_argument("--auto", action="store_true",
                    help="record it as an automatic pick (how: auto), not a selection")
    ap.add_argument("--list", action="store_true", help="show the folder and exit")
    a = ap.parse_args(argv)
    if a.list or not a.run_dir:
        return list_entries()
    run = Path(a.run_dir).resolve()
    ck = run / "checkpoints"
    if not ck.is_dir():
        raise SystemExit(f"[fail] {run} has no checkpoints/ folder")
    if a.ckpt:
        target = Path(a.ckpt).resolve()
    elif a.epoch is not None:
        hits = sorted(ck.glob(f"epoch={a.epoch:04d}-*.ckpt")) or sorted(ck.glob(f"epoch={a.epoch:03d}-*.ckpt"))
        if not hits:
            raise SystemExit(f"[fail] no epoch {a.epoch} checkpoint in {ck}: "
                             f"{[p.name for p in sorted(ck.glob('epoch=*.ckpt'))]}")
        target = hits[-1].resolve()
    else:
        raise SystemExit("[fail] give --epoch N or --ckpt PATH")
    if not target.is_file():
        raise SystemExit(f"[fail] not a file: {target}")
    if not a.auto and not a.why:
        raise SystemExit("[fail] --why is required for a selection (or pass --auto)")
    bc = _best_ckpt()
    how = "auto: operator-run publish (no held-out evidence given)" if a.auto \
        else f"selected: {a.why}"
    link = bc.publish(BEST_DIR, run, target, how=how, overwrite_selected=True,
                      extra={"published_by": "tools/publish_ckpt.py"})
    # keep the run's own selected.ckpt in step (the older spelling)
    sel = ck / "selected.ckpt"
    if sel.is_symlink() or sel.exists():
        sel.unlink()
    os.symlink(target.name, sel)
    m = re.search(r"epoch=(\d+)", target.name)
    print(f"published {link.relative_to(REPO)} -> {os.readlink(link)}\n"
          f"  epoch {m.group(1) if m else '?'}  {how}\n"
          f"  {sel.relative_to(REPO)} -> {target.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
