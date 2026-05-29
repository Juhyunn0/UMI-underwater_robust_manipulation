#!/usr/bin/env python3
"""replay_survey.py — read a survey run's diagnostic "black box" and print a
one-page summary.

Usage:
    python tools/replay_survey.py data/20260528/20260528_143000_survey/

This is intentionally a STUB. Right now it only prints a text summary from the
diagnostic CSV / JSON files written by survey_diagnostics.SurveyDiagnosticsRecorder
(no plotting). The plotting / animation work is deferred — see the TODO block.

Stdlib-only by design so it runs anywhere a recording lands.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# TODO (deferred — flesh out separately, do NOT build now):
#   1. Animated GIF of tag-map evolution from tag_snapshots/*.yaml
#      (one frame per snapshot; color tags by uncertainty; mark batch events).
#   2. Time-series plot: median_residual_px, last_jump_residual_mm, tags_in_graph
#      vs elapsed_s, with vertical markers at every batch_events.csv row.
#   3. Plot: anchor_position_drift_from_t0_mm (+ rotation drift) over time from
#      anchor_stability.csv — should stay ~0; a ramp means the datum is moving.
#   4. Per-tag position-vs-time from tag_history.csv "observation"/"shifted"
#      events — catch the exact moment a tag's estimate destabilizes.
#   5. Markdown report interleaving user_notes.csv inline with the timeline so a
#      human annotation ("map starting to squeeze ←") sits next to the metrics.
#   6. Cross-reference drift_diagnostics.approximate_drift_onset_t_s against the
#      nearest user note + batch event to attribute cause.
# ──────────────────────────────────────────────────────────────────────────────


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rule(title: str = "") -> str:
    if not title:
        return "─" * 72
    return f"── {title} " + "─" * max(0, 72 - len(title) - 4)


def summarize(run_dir: Path) -> int:
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory", file=sys.stderr)
        return 2

    summary = _read_json(run_dir / "diagnostics_summary.json")
    frames = _read_csv(run_dir / "survey_diagnostics.csv")
    batches = _read_csv(run_dir / "batch_events.csv")
    notes = _read_csv(run_dir / "user_notes.csv")
    actions = _read_csv(run_dir / "user_actions.csv")
    loops = _read_csv(run_dir / "loop_closure_events.csv")
    snaps = sorted((run_dir / "tag_snapshots").glob("snapshot_*.yaml")) \
        if (run_dir / "tag_snapshots").is_dir() else []

    print(_rule("RUN"))
    meta = summary.get("run_metadata", {})
    for k in ("anchor_tag_id", "survey_duration_s", "n_frames_processed",
              "n_tags_qualified", "git_commit", "tool_version", "fisheye_calib_path"):
        if k in meta:
            print(f"  {k:22s}: {meta[k]}")

    print(_rule("FILES"))
    print(f"  survey_diagnostics rows : {len(frames)}")
    print(f"  batch_events            : {len(batches)}")
    print(f"  loop_closure_events     : {len(loops)}")
    print(f"  user_actions            : {len(actions)}")
    print(f"  user_notes              : {len(notes)}")
    print(f"  tag_snapshots           : {len(snaps)}")

    print(_rule("SUMMARY STATS"))
    for k, v in summary.get("summary_stats", {}).items():
        print(f"  {k:26s}: {v}")

    print(_rule("DRIFT DIAGNOSTICS"))
    for k, v in summary.get("drift_diagnostics", {}).items():
        print(f"  {k:32s}: {v}")

    print(_rule("TAGS"))
    print(f"  {'tag':>5}  {'first_seen_s':>12}  {'promoted_s':>10}  "
          f"{'n_obs':>7}  {'final_unc_mm':>12}")
    for tid, rec in summary.get("tag_summary", {}).items():
        print(f"  {tid:>5}  {str(rec.get('first_seen_t_s')):>12}  "
              f"{str(rec.get('promoted_t_s')):>10}  {str(rec.get('n_obs')):>7}  "
              f"{str(rec.get('final_unc_mm')):>12}")

    if notes:
        print(_rule("USER NOTES (timeline)"))
        for n in notes:
            try:
                el = float(n.get("elapsed_s", 0.0))
                mmss = f"{int(el)//60:02d}:{int(el)%60:02d}"
            except (TypeError, ValueError):
                mmss = "--:--"
            print(f"  [{mmss}] (frame {n.get('frame_idx')}): {n.get('note_text')}")

    print(_rule())
    print("This is a stub. See the TODO block at the top of this file for the "
          "plots/animations we still want.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a data/.../<ts>_survey/ recording folder")
    args = ap.parse_args(argv)
    return summarize(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
