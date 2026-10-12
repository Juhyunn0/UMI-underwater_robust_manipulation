#!/usr/bin/env python3
"""mpc_tune.py — save / list / show MPC_Tuned profiles (control/mpc_tune.py).

    python -m rov_gui.tools.mpc_tune save NAME --description "what this tune is for"
    python -m rov_gui.tools.mpc_tune list
    python -m rov_gui.tools.mpc_tune show NAME          # + how it differs from now
    python -m rov_gui.tools.mpc_tune diff A B

``save`` snapshots what mpc_tuned FLIES at launch into
config/mpc_tuned_profiles/NAME.yaml:
  * by default the profile hw_mpc.yaml ``mpc_tuned_profile`` names (its
    ``apply:`` verbatim — editing hw_mpc.yaml's blocks changes NOTHING under
    mpc_tuned while a profile is held, so saving them would record values
    that never flew; safety review 2026-10-04);
  * ``--from-config``: hw_mpc.yaml's own blocks (``mpc_tuned:``,
    ``policy.q_scale`` / ``along_scale``, ``vehicle_net_buoyancy_n``,
    ``axis_cap``, ``axis_slew_per_s``) + params.py Q/QN/R — what mpc_tuned
    flies under ``--mpc-tune none``;
  * ``--from PROFILE``: a copy of another profile's ``apply:``.
Workflows for a new tune: (a) ``save NEW --from OLD``, edit NEW.yaml's
``apply:``, pick it on the panel; or (b) edit hw_mpc.yaml, fly with
``--mpc-tune none``, then ``save NEW --from-config``. Nothing here builds a
solver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import mpc_tune as MT                       # noqa: E402


def _station(config: str):
    """(cfg, P) as a station launched with ``config`` would hold them: the
    params module with the plant overrides applied (what the solver is
    generated from)."""
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.mpc_bridge import apply_plant_overrides, import_dobmpc

    cfg = MpcConfig.load(config)
    P = import_dobmpc(cfg.rov_model)["P"]
    apply_plant_overrides(P, cfg.plant)
    return cfg, P


def _flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _diff(a: dict, b: dict) -> list[str]:
    fa, fb = _flat(a), _flat(b)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k, "—"), fb.get(k, "—")
        literal = (str, bool, type(None))
        same = (va == vb if isinstance(va, literal) or isinstance(vb, literal)
                else MT._close(va, vb))
        if same:
            continue
        rows.append(f"  {k:28s} {MT._short(va):>32s}  →  {MT._short(vb)}")
    return rows


def cmd_save(a) -> int:
    cfg, P = _station(a.config)
    if a.from_config and a.from_profile:
        raise SystemExit("--from-config and --from are exclusive")
    if a.from_profile:
        base = MT.load_profile(a.from_profile)
    elif a.from_config or MT.is_no_profile(cfg.mpc_tuned_profile):
        base = None
    else:
        base = MT.load_profile(cfg.mpc_tuned_profile)
    print("apply: from " + (f"profile {base.name} (what mpc_tuned flies)"
                            if base is not None else
                            f"{a.config} blocks + params.py"
                            + ("" if MT.is_no_profile(cfg.mpc_tuned_profile)
                               else f" — NOTE mpc_tuned flies these only "
                                    f"under --mpc-tune none (the config "
                                    f"holds {cfg.mpc_tuned_profile})")))
    d = MT.snapshot(cfg, P, name=a.name, description=a.description,
                    config_path=a.config, base=base)
    path = MT.write_profile(d, MT.resolve_path(a.name), overwrite=a.overwrite)
    prof = MT.load_profile(path)
    print(f"saved {path}  (sha1 {prof.sha1[:12]})")
    for k, v in prof.apply.items():
        print(f"  apply.{k:24s} {v}")
    return 0


def cmd_list(_a) -> int:
    rows = MT.list_profiles()
    if not rows:
        print(f"no profiles in {MT.PROFILE_DIR}")
    for r in rows:
        tag = f"  ERROR {r['error']}" if r["error"] else ""
        print(f"{r['name']:32s} {r['created'][:19]:19s}  "
              f"{r['description'][:60]}{tag}")
    return 0


def cmd_show(a) -> int:
    prof = MT.load_profile(a.name)
    print(f"{prof.name}  ({prof.path})\n  sha1 {prof.sha1}\n  created "
          f"{prof.created}\n  {prof.description}")
    for k, v in prof.apply.items():
        print(f"  apply.{k:24s} {v}")
    cfg, P = _station(a.config)
    rows = _diff(prof.apply, MT.current_apply(cfg, P))
    print(f"\napply: profile → {a.config}'s own blocks (PID/mpc read these; "
          f"mpc_tuned only under --mpc-tune none)"
          + ("" if rows else ": identical"))
    print("\n".join(rows))
    warns = MT.profile_warnings(prof, P=P, cfg=cfg)
    print("\nwarnings on this station:" + ("" if warns else " none"))
    for w in warns:
        print(f"  - {w}")
    return 0


def cmd_diff(a) -> int:
    pa, pb = MT.load_profile(a.a), MT.load_profile(a.b)
    rows = (_diff(pa.apply, pb.apply)
            + _diff({"compare": pa.compare}, {"compare": pb.compare}))
    print(f"{pa.name} → {pb.name}" + ("" if rows else ": identical"))
    print("\n".join(rows))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/hw_mpc.yaml",
                   help="the station config to snapshot / compare against")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save", help="snapshot the current mpc_tuned tuning")
    s.add_argument("name")
    s.add_argument("--description", default="")
    s.add_argument("--from-config", action="store_true",
                   help="take apply: from hw_mpc.yaml's blocks + params.py "
                        "(what mpc_tuned flies under --mpc-tune none)")
    s.add_argument("--from", dest="from_profile", default=None,
                   metavar="PROFILE", help="copy another profile's apply:")
    s.add_argument("--overwrite", action="store_true",
                   help="replace an existing profile (run metas pin its sha1 "
                        "— prefer a new name)")
    sub.add_parser("list", help="the profiles in config/mpc_tuned_profiles/")
    s = sub.add_parser("show", help="one profile + its difference from now")
    s.add_argument("name")
    s = sub.add_parser("diff", help="two profiles")
    s.add_argument("a")
    s.add_argument("b")
    a = p.parse_args(argv)
    return {"save": cmd_save, "list": cmd_list, "show": cmd_show,
            "diff": cmd_diff}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
