#!/usr/bin/env python3
"""migrate_data_layout_20260914.py — fold ``sessions/`` and ``data/outputs/``
into the one dated tree ``data/YYYYMMDD/<run>[_kind]``.

Operator decision 2026-09-14: one root, dated, and the old citations in
config / docs / journal / memory are REWRITTEN to the new paths (no symlink
shim). This script is the record of exactly how every folder moved; the same
mapping is written to ``data/MIGRATION_20260914.json`` so a reader who finds
an old path in a record that could not be rewritten (a ``_pose.jsonl`` MESH
row, a ``selected.json`` evidence path, an old log) can still resolve it.

Layout after the move (``rov_gui/runstore.py`` is the code-side contract)::

    data/
      YYYYMMDD/
        MMDD_HHMMSS                the WATER run tree, as before (leaf unchanged)
        MMDD_HHMMSS_observe        was sessions/policy_observe/<day>/<leaf>
        MMDD_HHMMSS_landdry        was sessions/land_dryruns/<day>/<leaf>
        MMDD_HHMMSS_dryrun         was sessions/dryruns/dryrun_*/<day>/<leaf>
        MMDD_HHMMSS_obj            was sessions/pose_meshes/obj_<stamp>
        MMDD_HHMMSS/nav_HHMMSS/    was sessions/nav_runs/<stamp>/  (pre-runstore)
        MMDD_HHMMSS/<stamp>.csv    was sessions/mpc_runs/<stamp>.csv (pre-runstore)
        MMDD_HHMMSS/c3_depth_*.*   was sessions/ui_recordings/c3_depth_<stamp>*
        demonstration_NNNN         was sessions/demonstration_NNNN (umi_handheld)
        MMDD_HHMMSS_train_<task>_<exp>   was data/outputs/Y.M.D/H.M.S_<name>_<task>
        train_logs/                was data/outputs/train_logs (ablation stdout)
        YYYYMMDD_HHMMSS_<suffix>   the gantry/ZED/survey folders, untouched
      checkpoints/                 one flat folder: the best ckpt of every
                                   training run (symlink + .json sidecar)

The pre-runstore flat trees (nav_runs / mpc_runs / ui_recordings) are grouped
into one run folder per START the same way ``runstore.run_dir`` joins live
writers: artefacts whose stamps chain within JOIN_WINDOW_S (90 s) share the
folder named after the earliest of them.

Usage::

    python tools/migrate_data_layout_20260914.py            # dry run: print the plan
    python tools/migrate_data_layout_20260914.py --apply    # move + write manifest
    python tools/migrate_data_layout_20260914.py --citations  # print old->new rules

Every move is ``os.rename`` on one filesystem (no copy, no data touched).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SESSIONS = REPO / "sessions"
OUTPUTS = DATA / "outputs"
JOIN_WINDOW_S = 90.0

_STAMP = re.compile(r"(\d{8})_(\d{6})")


def _stamp_t(stamp: str) -> float:
    return time.mktime(time.strptime(stamp, "%Y%m%d_%H%M%S"))


def _leaf(stamp: str, kind: str = "") -> str:
    """``YYYYMMDD_HHMMSS`` -> ``MMDD_HHMMSS[_kind]``."""
    day, hms = stamp.split("_")
    return f"{day[4:]}_{hms}" + (f"_{kind}" if kind else "")


def _day(stamp: str) -> str:
    return stamp.split("_")[0]


# ------------------------------------------------------------------ plan
class Plan:
    def __init__(self):
        self.moves: list[tuple[Path, Path]] = []     # (src, dst) both absolute
        self.notes: list[str] = []

    def mv(self, src: Path, dst: Path, note: str = ""):
        assert src.exists(), f"missing source: {src}"
        self.moves.append((src, dst))
        if note:
            self.notes.append(note)

    def rel(self, p: Path) -> str:
        return str(p.relative_to(REPO))


def plan_llc(plan: Plan):
    """sessions/low_level_controller_data/<day>/<leaf> -> data/<day>/<leaf>."""
    base = SESSIONS / "low_level_controller_data"
    for day in sorted(base.iterdir()):
        if not day.is_dir():
            continue
        for leaf in sorted(day.iterdir()):
            plan.mv(leaf, DATA / day.name / leaf.name)


def plan_kind_tree(plan: Plan, tree: str, kind: str):
    """sessions/<tree>/<day>/<leaf> -> data/<day>/<leaf>_<kind>."""
    base = SESSIONS / tree
    if not base.is_dir():
        return
    for day in sorted(base.iterdir()):
        if not day.is_dir():
            continue
        for leaf in sorted(day.iterdir()):
            if not leaf.is_dir():
                plan.mv(leaf, DATA / day.name / leaf.name)   # stray file
                continue
            # policy_session_replay writes `<run>__replay_<name>` BESIDE the
            # run; the kind goes on the run part so the sibling relation
            # (`<run>_observe` + `<run>_observe__replay_<name>`) survives.
            m = re.match(r"^(.+?)(__replay_.+)$", leaf.name)
            if m:
                plan.mv(leaf, DATA / day.name / f"{m.group(1)}_{kind}{m.group(2)}")
            else:
                plan.mv(leaf, DATA / day.name / f"{leaf.name}_{kind}")


def plan_dryruns(plan: Plan):
    """sessions/dryruns/dryrun_<stamp>/<day>/<leaf> -> data/<day>/<leaf>_dryrun.

    The outer ``dryrun_<stamp>`` root was the per-invocation isolation trick;
    the INNER leaf is the run the writers actually joined, so that is the
    name that survives (they differ by a few seconds).
    """
    base = SESSIONS / "dryruns"
    if not base.is_dir():
        return
    for root in sorted(base.iterdir()):
        if not root.is_dir():
            continue
        for day in sorted(root.iterdir()):
            if not day.is_dir():
                continue
            for leaf in sorted(day.iterdir()):
                plan.mv(leaf, DATA / day.name / f"{leaf.name}_dryrun")


def plan_pose_meshes(plan: Plan):
    """sessions/pose_meshes/obj_<stamp>[_n] -> data/<day>/MMDD_HHMMSS[_n]_obj."""
    base = SESSIONS / "pose_meshes"
    if not base.is_dir():
        return
    for d in sorted(base.iterdir()):
        m = re.match(r"^obj_(\d{8}_\d{6})(?:_(\d+))?$", d.name)
        if not m:
            plan.notes.append(f"pose_meshes: unexpected entry left in place: {d}")
            continue
        stamp, n = m.group(1), m.group(2)
        leaf = _leaf(stamp) + (f"_{n}" if n else "") + "_obj"
        plan.mv(d, DATA / _day(stamp) / leaf)


def plan_flat_prerunstore(plan: Plan):
    """nav_runs / mpc_runs / ui_recordings -> one run folder per START.

    Artefacts are ordered by stamp; each joins the open group when it is
    within JOIN_WINDOW_S of the previous artefact, else starts a new group
    named after its own stamp (runstore's join rule, applied offline).
    """
    items: list[tuple[str, str, Path]] = []      # (stamp, kind, path)
    nav = SESSIONS / "nav_runs"
    if nav.is_dir():
        for d in sorted(nav.iterdir()):
            m = re.fullmatch(r"(\d{8}_\d{6})", d.name)
            if m and d.is_dir():
                items.append((m.group(1), "nav", d))
            else:
                plan.notes.append(f"nav_runs: unexpected entry left in place: {d}")
    mpc = SESSIONS / "mpc_runs"
    if mpc.is_dir():
        for f in sorted(mpc.iterdir()):
            m = re.match(r"^(\d{8}_\d{6})", f.name)
            if m:
                items.append((m.group(1), "mpc", f))
            elif f.name == "events.log":
                # ONE flat engage/refusal log for every pre-runstore run
                # (2026-08-13..14); filed under the day of its first line,
                # renamed so its origin stays readable.
                first = f.read_text(errors="replace").split("\n", 1)[0]
                mm = re.match(r"(\d{4})-(\d{2})-(\d{2})", first)
                day = "".join(mm.groups()) if mm else "20260813"
                plan.mv(f, DATA / day / "mpc_runs_events.log",
                        note="sessions/mpc_runs/events.log -> data/%s/mpc_runs_events.log "
                             "(flat pre-runstore events file, all runs)" % day)
            else:
                plan.notes.append(f"mpc_runs: unexpected entry left in place: {f}")
    ui = SESSIONS / "ui_recordings"
    if ui.is_dir():
        for f in sorted(ui.iterdir()):
            m = _STAMP.search(f.name)
            if m:
                items.append((f"{m.group(1)}_{m.group(2)}", "ui", f))
            else:
                plan.notes.append(f"ui_recordings: unexpected entry left in place: {f}")
    items.sort(key=lambda it: (it[0], it[1], it[2].name))
    group_stamp, last_t = None, None
    for stamp, kind, path in items:
        t = _stamp_t(stamp)
        if group_stamp is None or _day(stamp) != _day(group_stamp) \
                or (t - last_t) > JOIN_WINDOW_S:
            group_stamp = stamp
        last_t = t
        run = DATA / _day(group_stamp) / _leaf(group_stamp)
        if kind == "nav":
            plan.mv(path, run / f"nav_{stamp.split('_')[1]}")
        else:
            plan.mv(path, run / path.name)


def plan_demos(plan: Plan):
    """sessions/<kind>_NNNN (umi_handheld takes) -> data/<mtime day>/<same>."""
    for d in sorted(SESSIONS.iterdir()):
        if d.is_dir() and re.fullmatch(r"(demonstration|grippercalibration|session)_\w+", d.name):
            day = time.strftime("%Y%m%d", time.localtime(d.stat().st_mtime))
            plan.mv(d, DATA / day / d.name,
                    note=f"{d.name}: dated by folder mtime ({day})")


def training_run_name(run: Path) -> tuple[str, str]:
    """(day, leaf) for data/outputs/Y.M.D/H.M.S_<name>_<task>: reads exp_name
    and task name from the run's .hydra/config.yaml (no yaml dependency —
    both are plain top-level scalars)."""
    day = run.parent.name.replace(".", "")
    hms = run.name.split("_", 1)[0].replace(".", "")
    exp, task = "", ""
    cfgp = run / ".hydra" / "config.yaml"
    if cfgp.is_file():
        for line in cfgp.read_text().splitlines():
            if line.startswith("exp_name:"):
                exp = line.split(":", 1)[1].strip().strip("'\"")
            elif re.match(r"^task:\s*$", line):
                pass
    # task.name lives one level down; fall back to the folder's tail.
    m = re.search(r"_(umi_[a-z0-9_]+)$", run.name)
    task = m.group(1) if m else run.name.split("_", 1)[1]
    if cfgp.is_file():
        txt = cfgp.read_text()
        mm = re.search(r"^task:\n(?:.*\n)*?\s+name:\s*(\S+)", txt, re.M)
        if mm:
            task = mm.group(1).strip("'\"")
    leaf = f"{day[4:]}_{hms}_train_{task}" + (f"_{exp}" if exp else "")
    return day, leaf


def plan_outputs(plan: Plan):
    if not OUTPUTS.is_dir():
        return
    for dated in sorted(OUTPUTS.iterdir()):
        if dated.name == "train_logs":
            for f in sorted(dated.iterdir()):
                # the ablation runner stamps its logs (NAME_MMDD_HHMMSS.log);
                # a run that crossed midnight is filed under the day it began
                mtime_day = time.strftime("%Y%m%d", time.localtime(f.stat().st_mtime))
                m = re.search(r"(\d{8})", f.name) or re.search(r"_(\d{4})_\d{6}", f.name)
                day = (m.group(1) if m and len(m.group(1)) == 8
                       else (mtime_day[:4] + m.group(1)) if m else mtime_day)
                plan.mv(f, DATA / day / "train_logs" / f.name)
            continue
        if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", dated.name):
            plan.notes.append(f"outputs: unexpected entry left in place: {dated}")
            continue
        for run in sorted(dated.iterdir()):
            if not run.is_dir():
                plan.notes.append(f"outputs: stray file left in place: {run}")
                continue
            day, leaf = training_run_name(run)
            plan.mv(run, DATA / day / leaf)


# ------------------------------------------------------------------ best ckpts
def best_ckpt_entries(moves: list[tuple[Path, Path]]) -> list[dict]:
    """For every training run: the checkpoint to expose in data/checkpoints/.

    ``selected.ckpt`` (a held-out selection, see selected.json) wins; else the
    top-1 by the run's monitor key (train_loss, mode min — the number in the
    file name). Recorded in the sidecar so nobody mistakes an automatic pick
    for a selection.
    """
    out = []
    for src, dst in moves:
        m = re.fullmatch(r"(\d{4})_(\d{6})_train_(.+)", dst.name)
        if not m or not (src / "checkpoints").is_dir():
            continue
        ck = src / "checkpoints"
        stamp = f"{dst.parent.name}_{m.group(2)}"
        name = f"{stamp}_{m.group(3)}"
        sel = ck / "selected.ckpt"
        if sel.is_symlink() or sel.exists():
            target = Path(os.readlink(sel)) if sel.is_symlink() else sel
            target = (ck / target).resolve() if not target.is_absolute() else target
            how = "selected: held-out replay (see selected.json beside it)"
        else:
            cands = []
            for f in ck.glob("epoch=*.ckpt"):
                mm = re.search(r"train_loss=([0-9]+\.[0-9]+)", f.name)
                if mm:
                    cands.append((float(mm.group(1)), f))
            if not cands:
                continue
            cands.sort(key=lambda c: (c[0], c[1].name))
            target = cands[0][1]
            how = "auto: top-1 by train_loss (mode min) among the run's topk checkpoints"
        ep = re.search(r"epoch=(\d+)", target.name)
        out.append({
            "name": name,
            "run_dir": str(dst.relative_to(REPO)),
            "target": str((dst / "checkpoints" / target.name).relative_to(REPO)),
            "epoch": int(ep.group(1)) if ep else None,
            "how": how,
        })
    return out


def write_best_ckpts(entries: list[dict], apply: bool):
    root = DATA / "checkpoints"
    for e in entries:
        link = root / f"{e['name']}.ckpt"
        target = REPO / e["target"]
        rel = os.path.relpath(target, root)
        side = {
            "target": e["target"], "run_dir": e["run_dir"], "epoch": e["epoch"],
            "how": e["how"], "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": ("symlink, not a copy: `ls -l` shows the epoch file it points "
                     "at; the run folder keeps every topk checkpoint as before"),
        }
        print(f"  ckpt  {link.relative_to(REPO)} -> {rel}   [{e['how'].split(':')[0]}]")
        if apply:
            root.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(rel, link)
            link.with_suffix(".json").write_text(json.dumps(side, indent=2,
                                                            ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ citations
def citation_rules(moves: list[tuple[Path, Path]]) -> list[tuple[str, str]]:
    """Ordered (old, new) TEXT substitutions for the repo's citations.

    Longest/most specific first, so ``sessions/policy_observe/20260908/0908_180453``
    is rewritten before the bare tree name catches it. Bare tree names map to
    a glob that still reads as "that kind of run".
    """
    rules: list[tuple[str, str]] = []
    for src, dst in moves:
        s, d = str(src.relative_to(REPO)), str(dst.relative_to(REPO))
        rules.append((s, d))
        # dryruns were cited by their OUTER per-invocation root as well
        m = re.match(r"^sessions/dryruns/(dryrun_\d{8}_\d{6})/\d{8}/(\d{4}_\d{6})$", s)
        if m:
            rules.append((f"sessions/dryruns/{m.group(1)}", d))
    rules.sort(key=lambda r: -len(r[0]))
    # Per-DAY prefixes of the kind trees and the flat pre-runstore trees
    # (a citation that names the day and then a placeholder or a wrapped
    # leaf): the day directory is the same, only the leaf changed.
    days: dict[str, set] = {}
    for src, dst in moves:
        s = str(src.relative_to(REPO))
        m = re.match(r"^sessions/(policy_observe|land_dryruns|low_level_controller_data)/(\d{8})/", s)
        if m:
            days.setdefault(m.group(1), set()).add(m.group(2))
        m = re.match(r"^sessions/(mpc_runs|nav_runs|ui_recordings)/[a-z0-9_]*?(\d{8})_", s)
        if m:
            days.setdefault(m.group(1), set()).add(m.group(2))
        m = re.match(r"^data/outputs/(\d{4})\.(\d{2})\.(\d{2})/", s)
        if m:
            days.setdefault("outputs", set()).add("".join(m.groups()))
    day_rules = []
    for tree, ds in days.items():
        for d in sorted(ds):
            if tree in ("policy_observe", "land_dryruns", "low_level_controller_data"):
                day_rules.append((f"sessions/{tree}/{d}/", f"data/{d}/"))
            elif tree == "ui_recordings":
                # one file family per START; a partial stem is caught by the
                # rewriter's own stem rules, this is the bare-day fallback
                day_rules.append((f"sessions/{tree}/", "data/*/*/"))
            elif tree == "outputs":
                day_rules.append((f"data/outputs/{d[:4]}.{d[4:6]}.{d[6:]}/", f"data/{d}/"))
            else:                                   # mpc_runs / nav_runs stamps
                day_rules.append((f"sessions/{tree}/{d}_", f"data/{d}/{d[4:]}_"))
    for src, dst in moves:
        s_ = str(src.relative_to(REPO))
        m = re.match(r"^data/outputs/(\d{4}\.\d{2}\.\d{2})/(\d{2}\.\d{2}\.\d{2})_", s_)
        if m:
            d_ = str(dst.relative_to(REPO))
            day = m.group(1).replace(".", "")
            day_rules += [(f"data/outputs/{m.group(1)}/{m.group(2)}", d_),
                          (f"data/{day}/{m.group(2)}", d_),
                          (f"{m.group(1)}/{m.group(2)}", d_)]
    rules += sorted(set(day_rules), key=lambda r: -len(r[0]))
    # Placeholder spellings used in docs/docstrings, then the bare tree
    # names — each mapped to the glob that still reads as "that kind of run".
    rules += [
        ("sessions/nav_runs/20260813_17*", "data/20260813/0813_17*/nav_*"),
        ("sessions/nav_runs/<stamp>/", "data/YYYYMMDD/MMDD_HHMMSS/nav_<hhmmss>/"),
        ("sessions/nav_runs/*", "data/*/*/nav_*"),
        ("sessions/nav_runs", "data/*/*/nav_*"),
        ("sessions/mpc_runs/<stamp>.csv", "data/YYYYMMDD/MMDD_HHMMSS/<stamp>.csv"),
        ("sessions/mpc_runs", "data/*/*"),
        ("sessions/ui_recordings/c3_depth_<stamp>*", "data/YYYYMMDD/MMDD_HHMMSS/c3_depth_<stamp>*"),
        ("sessions/ui_recordings", "data/*/*"),
        ("sessions/pose_meshes/obj_<stamp>[_n]", "data/YYYYMMDD/MMDD_HHMMSS[_n]_obj"),
        ("sessions/pose_meshes/obj_<stamp>", "data/YYYYMMDD/MMDD_HHMMSS_obj"),
        ("sessions/pose_meshes/obj_<시각>/", "data/<날짜>/<시각>_obj/"),
        ("sessions/pose_meshes/*", "data/*/*_obj"),
        ("sessions/pose_meshes/", "data/*/*_obj/"),
        ("sessions/pose_meshes", "data/*/*_obj"),
        ("sessions/policy_observe/<날짜>/<시각>/", "data/<날짜>/<시각>_observe/"),
        ("sessions/policy_observe/<날짜>/<시각>", "data/<날짜>/<시각>_observe"),
        ("sessions/policy_observe/<date>/<hhmmss>/", "data/<date>/<hhmmss>_observe/"),
        ("sessions/policy_observe/<day>/<leaf>", "data/<day>/<leaf>_observe"),
        ("sessions/policy_observe/", "data/*/*_observe/"),
        ("sessions/policy_observe", "data/*/*_observe"),
        ("sessions/land_dryruns/<날짜>/<시각>/", "data/<날짜>/<시각>_landdry/"),
        ("sessions/land_dryruns/<day>/<leaf>", "data/<day>/<leaf>_landdry"),
        ("sessions/land_dryruns/", "data/*/*_landdry/"),
        ("sessions/land_dryruns", "data/*/*_landdry"),
        ("sessions/dryruns/", "data/*/*_dryrun/"),
        ("sessions/dryruns", "data/*/*_dryrun"),
        ("sessions/low_level_controller_data/", "data/"),
        ("sessions/low_level_controller_data", "data"),
        ("data/outputs/train_logs", "data/<YYYYMMDD>/train_logs"),
        ("data/outputs/", "data/<YYYYMMDD>/*_train_*/"),
        ("data/outputs", "data/<YYYYMMDD>/*_train_*"),
    ]
    return rules


# ------------------------------------------------------------------ main
def build_plan() -> Plan:
    plan = Plan()
    plan_llc(plan)
    plan_kind_tree(plan, "policy_observe", "observe")
    plan_kind_tree(plan, "land_dryruns", "landdry")
    plan_dryruns(plan)
    plan_pose_meshes(plan)
    plan_flat_prerunstore(plan)
    plan_demos(plan)
    plan_outputs(plan)
    # collisions
    seen: dict[Path, Path] = {}
    for src, dst in plan.moves:
        assert dst not in seen, f"destination collision: {dst} <- {src} and {seen[dst]}"
        assert not dst.exists(), f"destination already exists: {dst} (from {src})"
        seen[dst] = src
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="perform the moves")
    ap.add_argument("--citations", action="store_true",
                    help="print the old->new text rules and exit")
    a = ap.parse_args(argv)
    manifest = DATA / "MIGRATION_20260914.json"
    if a.citations and manifest.is_file():
        # After the move the sources are gone: the rules come from the record.
        rec = json.loads(manifest.read_text())
        moves = [(REPO / m["from"], REPO / m["to"]) for m in rec["moves"]]
        for old, new in citation_rules(moves):
            print(f"{old}\t{new}")
        return 0
    plan = build_plan()
    if a.citations:
        for old, new in citation_rules(plan.moves):
            print(f"{old}\t{new}")
        return 0
    print(f"{len(plan.moves)} moves")
    for src, dst in plan.moves:
        print(f"  {plan.rel(src)}\n      -> {plan.rel(dst)}")
    for n in plan.notes:
        print(f"  note: {n}")
    entries = best_ckpt_entries(plan.moves)
    print(f"{len(entries)} best-checkpoint entries for data/checkpoints/")
    write_best_ckpts(entries, apply=False)
    if not a.apply:
        print("\n(dry run — pass --apply to move)")
        return 0
    manifest = {
        "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layout": "data/YYYYMMDD/<run>[_kind]  (rov_gui/runstore.py)",
        "moves": [{"from": plan.rel(s), "to": plan.rel(d)} for s, d in plan.moves],
        "best_checkpoints": entries,
        "notes": plan.notes,
        "script": str(Path(__file__).relative_to(REPO)),
    }
    for src, dst in plan.moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
    write_best_ckpts(entries, apply=True)
    (DATA / "MIGRATION_20260914.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    # drop the now-empty scaffolding (never anything with content)
    for root in (SESSIONS, OUTPUTS):
        if root.is_dir():
            for d in sorted((p for p in root.rglob("*") if p.is_dir()),
                            key=lambda p: -len(p.parts)):
                try:
                    d.rmdir()
                except OSError:
                    pass
            try:
                root.rmdir()
            except OSError:
                print(f"  left in place (not empty): {root.relative_to(REPO)}")
    print(f"done — manifest data/MIGRATION_20260914.json ({len(plan.moves)} moves)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
