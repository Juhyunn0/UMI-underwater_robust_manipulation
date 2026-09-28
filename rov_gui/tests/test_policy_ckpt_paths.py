#!/usr/bin/env python3
"""
test_policy_ckpt_paths.py — ONE checkpoint path, everywhere (no torch, no Qt).

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_policy_ckpt_paths.py

EXPECTED TO FAIL until the 5-dim retrain is selected. The station's policy
checkpoint is named in seven places — two YAML files and five Python
constants — and the 2026-09-07 action-contract change (10-dim pose10d ->
5-dim pos_yaw_width, ``umi_depth_5d``) makes a stale copy dangerous rather
than merely wrong: the old 10-dim network loads, passes every shape check,
and ``compose_plan`` would decode it by width. ``workers._policy_refusal``
refuses to arm on the contract, but the refusal message says "fix
policy.ckpt", and this test is what says WHICH copy. It asserts

* all seven name ONE string;
* that string contains ``umi_depth_5d`` (the retrain's task/run name);
* the file exists.

Today (2026-09-07) every copy still points at the 2026-09-01 10-dim run —
``rov_gui/state.py`` carries the ``TODO(5-dim retrain)`` — so the second and
third tests FAIL by design; they pass the moment the hub switches
``POLICY_CKPT_DEFAULT`` and the two YAML ``policy.ckpt`` lines to the selected
5-dim checkpoint. Do not weaken it to make it green.

Sources (all read WITHOUT importing torch or Qt):
    config/hw_mpc.yaml  policy.ckpt                     (yaml.safe_load)
    config/land_dp.yaml policy.ckpt                     (yaml.safe_load)
    rov_gui.state.POLICY_CKPT_DEFAULT                   (stdlib module)
    rov_gui.control.geometry.default_policy_block()["ckpt"]
    rov_gui/backends/policy.py _FALLBACK_POLICY_BLOCK["ckpt"]   (ast — the
        module pulls Qt through ..qt, so its text is parsed, not imported)
    rov_gui.perception.dp_policy.DEFAULT_CKPT           (torch is lazy there)
    rov_gui.tools.dp_policy_reference.DEFAULT_CKPT

Since 2026-09-11 the checkpoint named here is only the LAUNCH SEED: the
trajectory panel's picker (``PolicyWorker.set_ckpt``) can swap it at run
time, and the run record carries the loaded one (``ckpt_loaded`` /
``ckpt_source``). The single-source rule still matters for what loads before
anyone picks.

The same file also pins ``policy.max_run_s`` — ONE value in the four places
it is defined (both YAMLs, ``geometry.default_policy_block()`` and the
backend fallback block) — at the operator's decision
[결정: operator 2026-09-11 — 120 -> 500 s] (``test_max_run_s_*``).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml                                                    # noqa: E402

RETRAIN_TAG = "umi_depth_5d"


def _yaml_ckpt(rel: str) -> str:
    cfg = yaml.safe_load((ROOT / rel).read_text())
    blk = cfg.get("policy") if isinstance(cfg, dict) else None
    assert isinstance(blk, dict) and "ckpt" in blk, f"{rel}: no policy.ckpt"
    return str(blk["ckpt"])


def _fallback_block_ckpt() -> str:
    """``_FALLBACK_POLICY_BLOCK["ckpt"]`` from the module TEXT of
    rov_gui/backends/policy.py (it imports Qt at module scope). The value is
    either a string literal or a bare name imported from ``..state`` — the
    2026-09-07 form, ``"ckpt": POLICY_CKPT_DEFAULT`` — which is resolved
    against ``rov_gui.state`` only when the module really imports it from
    there (a same-named local constant would be a second source)."""
    src = (ROOT / "rov_gui" / "backends" / "policy.py").read_text()
    tree = ast.parse(src)
    from_state = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (
                (node.level == 2 and node.module == "state")
                or (node.level == 0 and node.module == "rov_gui.state")):
            from_state |= {(al.asname or al.name) for al in node.names}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "_FALLBACK_POLICY_BLOCK":
            assert isinstance(node.value, ast.Dict), "_FALLBACK_POLICY_BLOCK is not a dict literal"
            for k, v in zip(node.value.keys, node.value.values):
                if not (isinstance(k, ast.Constant) and k.value == "ckpt"):
                    continue
                if isinstance(v, ast.Name):
                    assert v.id in from_state, (
                        f"_FALLBACK_POLICY_BLOCK['ckpt'] = {v.id} is not imported "
                        f"from rov_gui.state — a second source, not a re-export")
                    from rov_gui import state
                    return str(getattr(state, v.id))
                return str(ast.literal_eval(v))
            raise AssertionError("_FALLBACK_POLICY_BLOCK has no 'ckpt' key")
    raise AssertionError("rov_gui/backends/policy.py: _FALLBACK_POLICY_BLOCK not found")


def _fallback_block_number(key: str) -> float:
    """``_FALLBACK_POLICY_BLOCK[key]`` (a numeric literal) from the module
    TEXT of rov_gui/backends/policy.py, like ``_fallback_block_ckpt``."""
    src = (ROOT / "rov_gui" / "backends" / "policy.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "_FALLBACK_POLICY_BLOCK":
            assert isinstance(node.value, ast.Dict), "_FALLBACK_POLICY_BLOCK is not a dict literal"
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == key:
                    return float(ast.literal_eval(v))
            raise AssertionError(f"_FALLBACK_POLICY_BLOCK has no {key!r} key")
    raise AssertionError("rov_gui/backends/policy.py: _FALLBACK_POLICY_BLOCK not found")


def _yaml_policy_number(rel: str, key: str) -> float:
    cfg = yaml.safe_load((ROOT / rel).read_text())
    blk = cfg.get("policy") if isinstance(cfg, dict) else None
    assert isinstance(blk, dict) and key in blk, f"{rel}: no policy.{key}"
    return float(blk[key])


#: [결정: operator 2026-09-11 — 120 -> 500 s]. Not a measurement: the operator
#: asked for a longer policy run clock; observe_max_run_s (1800) is unchanged.
MAX_RUN_S_DECISION = 500.0


def max_run_s_sources() -> dict:
    """source name -> policy.max_run_s, every place the station defines it."""
    from rov_gui.control import geometry
    return {
        "config/hw_mpc.yaml policy.max_run_s": _yaml_policy_number("config/hw_mpc.yaml", "max_run_s"),
        "config/land_dp.yaml policy.max_run_s": _yaml_policy_number("config/land_dp.yaml", "max_run_s"),
        "geometry.default_policy_block()['max_run_s']": float(geometry.default_policy_block()["max_run_s"]),
        "backends/policy.py _FALLBACK_POLICY_BLOCK['max_run_s']": _fallback_block_number("max_run_s"),
    }


def ckpt_sources() -> dict:
    """source name -> checkpoint path string, every place the station names one."""
    from rov_gui import state
    from rov_gui.control import geometry
    from rov_gui.perception import dp_policy
    from rov_gui.tools import dp_policy_reference
    return {
        "config/hw_mpc.yaml policy.ckpt": _yaml_ckpt("config/hw_mpc.yaml"),
        "config/land_dp.yaml policy.ckpt": _yaml_ckpt("config/land_dp.yaml"),
        "rov_gui.state.POLICY_CKPT_DEFAULT": str(state.POLICY_CKPT_DEFAULT),
        "geometry.default_policy_block()['ckpt']": str(geometry.default_policy_block()["ckpt"]),
        "backends/policy.py _FALLBACK_POLICY_BLOCK['ckpt']": _fallback_block_ckpt(),
        "dp_policy.DEFAULT_CKPT": str(dp_policy.DEFAULT_CKPT),
        "dp_policy_reference.DEFAULT_CKPT": str(dp_policy_reference.DEFAULT_CKPT),
    }


def _listing(src: dict) -> str:
    return "\n".join(f"      {k:50s} {v}" for k, v in src.items())


def test_all_sources_name_one_checkpoint():
    src = ckpt_sources()
    distinct = sorted(set(src.values()))
    assert len(distinct) == 1, (
        f"{len(distinct)} different checkpoint paths across {len(src)} sources — "
        f"the station would load one network and record another:\n{_listing(src)}")


def test_the_one_checkpoint_is_the_5dim_retrain():
    """FAILS by design until POLICY_CKPT_DEFAULT + the two YAML lines are
    switched to the selected umi_depth_5d checkpoint (see the header)."""
    src = ckpt_sources()
    stale = {k: v for k, v in src.items() if RETRAIN_TAG not in v}
    assert not stale, (
        f"{len(stale)}/{len(src)} sources still point at a checkpoint whose path "
        f"lacks {RETRAIN_TAG!r} (the 10-dim pose10d run; the station flies "
        f"pos_yaw_width and refuses to arm on it):\n{_listing(stale)}")


def test_the_checkpoint_file_exists():
    src = ckpt_sources()
    missing = {k: v for k, v in src.items() if not Path(v).is_file()}
    assert not missing, (
        f"{len(missing)}/{len(src)} sources name a checkpoint that is not on disk "
        f"(selected.ckpt not yet linked?):\n{_listing(missing)}")


def test_max_run_s_is_one_value_everywhere_and_is_the_operator_decision():
    """[결정: operator 2026-09-11 — 120 -> 500 s]: all four definitions agree
    and carry the decided value. A YAML that lags the code (or the reverse)
    would let a run fly one clock and record another."""
    src = max_run_s_sources()
    distinct = sorted(set(src.values()))
    assert len(distinct) == 1, (
        f"{len(distinct)} different max_run_s values across {len(src)} sources:\n"
        + "\n".join(f"      {k:55s} {v}" for k, v in src.items()))
    assert distinct[0] == MAX_RUN_S_DECISION, (
        f"policy.max_run_s is {distinct[0]}, the operator decided "
        f"{MAX_RUN_S_DECISION} (2026-09-11)")


# =============================================================================
# runner (same shape as test_policy_frames.py — works with or without pytest)
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
