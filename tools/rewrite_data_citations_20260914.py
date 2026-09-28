#!/usr/bin/env python3
"""rewrite_data_citations_20260914.py — point every old-path citation at the
folder it now lives in (the 2026-09-14 layout, tools/migrate_data_layout_20260914.py).

The measurement rule (CLAUDE.md): a number cited as measured names the
artifact path it came from, and that path must resolve. Moving 485 folders
would have turned ~300 citations in config comments, docs, the journal, the
memory index and code comments into dead references, so the citations are
rewritten with the SAME mapping the move used (``--citations`` of the
migration script), most specific rule first.

What is NOT rewritten, on purpose:

* ``external/`` — the vendored checkouts' own ``data/outputs`` are theirs;
* ``rov_gui/runstore.py`` and the two 2026-09-14 tools — they describe the
  OLD layout by design;
* any line containing ``2026-09-14`` — a historical note about the move;
* citations of folders that did not exist at migration time
  (``sessions/demonstration_0019``, ``sessions/grippercalibration_0004`` …)
  — they were dead before and are left as they were rather than pointed at
  a location that was never verified.

Usage::

    python tools/rewrite_data_citations_20260914.py            # dry run: counts per file
    python tools/rewrite_data_citations_20260914.py --apply
    python tools/rewrite_data_citations_20260914.py --apply --also data/*/*_train_*/checkpoints/selected.json
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEMORY = Path("/home/bdml/.claude/projects/-home-bdml-Desktop-umi-underwater-robust-control/memory")

SKIP_DIRS = {".git", "sessions", "data", ".cache", "refs", "__pycache__", "node_modules",
             "vendor", "_mpcc_gen", "_acados_gen", "external", "figures", "build_cache"}
SKIP_FILES = {
    REPO / "rov_gui" / "runstore.py",
    REPO / "tools" / "migrate_data_layout_20260914.py",
    REPO / "tools" / "rewrite_data_citations_20260914.py",
    REPO / ".gitignore",
}
#: Records the APP wrote at run time (a dryrun's meta.json, a bench report)
#: and generated pages: a record is not rewritten — data/MIGRATION_20260914.json
#: resolves the old path it carries.
SKIP_PREFIXES = (
    REPO / "rov_gui" / "tools" / "fstereo_bench_out",
    REPO / "rov_gui" / "tools" / "dp_policy_out",
    REPO / "rov_replay_0903_183405.html",
)
TEXT_EXT = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".sh", ".html", ".csv", ".cfg", ".toml", ".ini"}
SKIP_LINE = re.compile(r"2026-09-14")
#: the old hydra spelling of a training run ("2026.09.10/09.52.52", also after
#: a first pass turned the day part into "data/20260910/09.52.52")
_OLD_STAMP = re.compile(r"(?:\d{4}\.\d{2}\.\d{2}|data/\d{8})/\d{2}\.\d{2}\.\d{2}")


def rules() -> list[tuple[str, str]]:
    out = subprocess.run([sys.executable, str(REPO / "tools" / "migrate_data_layout_20260914.py"),
                          "--citations"], check=True, capture_output=True, text=True).stdout
    rs = []
    for line in out.splitlines():
        old, new = line.split("\t")
        rs.append((old, new))
    # a file's STEM (a sidecar family cited by its common prefix, e.g.
    # "sessions/ui_recordings/c3_depth_20260809_202409_" + a wrapped tail)
    extra = []
    for old, new in rs:
        o, n = Path(old), Path(new)
        if "." in o.name and n.name == o.name:
            stem = o.name.split(".")[0]
            extra.append((str(o.parent / stem), str(n.parent / stem)))
    seen = set(r[0] for r in rs)
    for r in extra:
        if r[0] not in seen:
            rs.append(r); seen.add(r[0])
    # most specific first; ties by original order
    rs.sort(key=lambda r: -len(r[0]))
    return rs


def candidate_files(extra: list[str]) -> list[Path]:
    files = []
    for root in (REPO, MEMORY):
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
                continue
            if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
                continue
            if p in SKIP_FILES or any(p == q or q in p.parents for q in SKIP_PREFIXES):
                continue
            files.append(p)
    for pat in extra:
        files += [p for p in REPO.glob(pat) if p.is_file()]
    return files


def rewrite_text(text: str, rs: list[tuple[str, str]]) -> tuple[str, int]:
    n = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if SKIP_LINE.search(line) or not ("sessions/" in line or "data/outputs" in line
                                          or _OLD_STAMP.search(line)):
            continue
        new = line
        for old, rep in rs:
            if old in new:
                new = new.replace(old, rep)
        if new != line:
            lines[i] = new
            n += 1
    return "\n".join(lines), n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--also", nargs="*", default=[], help="extra repo-relative globs")
    ap.add_argument("--show", action="store_true", help="print each changed line")
    a = ap.parse_args(argv)
    rs = rules()
    total = 0
    for p in sorted(set(candidate_files(a.also))):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new, n = rewrite_text(text, rs)
        if n:
            total += n
            print(f"{n:4d}  {p}")
            if a.show:
                for l1, l2 in zip(text.split("\n"), new.split("\n")):
                    if l1 != l2:
                        print(f"      - {l1.strip()[:150]}\n      + {l2.strip()[:150]}")
            if a.apply:
                p.write_text(new, encoding="utf-8")
    print(f"{total} lines {'rewritten' if a.apply else 'would be rewritten'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
