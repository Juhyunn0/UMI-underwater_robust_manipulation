#!/usr/bin/env python3
"""survey_diagnostics.py — "black box" recorder for a live tag-survey run.

A single ``SurveyDiagnosticsRecorder`` owns one daemon thread and one Queue.
Producers (the detection/SLAM worker thread AND the GUI thread) call ``log_*``
methods; each builds a small payload and enqueues it. The daemon consumes the
queue, formats CSV rows / YAML snapshots, writes them, updates the running
aggregates for the final summary, and flushes every ``flush_interval_s`` so a
crash loses at most ~1 s of data. The hot SLAM path never blocks on disk.

This module imports NO PyQt and no GTSAM — it is pure stdlib + numpy-free so the
headless CLI path stays unaffected. All callers pass already-extracted scalars.

File layout written under ``run_dir``:
    survey_diagnostics.csv     per-frame backend state
    tag_history.csv            sparse per-tag event log
    batch_events.csv           every periodic batch re-optimization
    user_actions.csv           every UI click / state transition
    anchor_stability.csv       anchor pose drift over time
    loop_closure_events.csv    camera revisit detections
    slam_internals.csv         iSAM2 health samples
    user_notes.csv             timestamped notes from the UI
    tag_snapshots/snapshot_tNNNs.yaml   periodic full tag-pose snapshots
    diagnostics_summary.json   summary stats, written at close()

UI labels, comments, and logs are English by project convention.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

# ── exact column lists (see the task spec; do not reorder without updating the
#    replay tool + README) ────────────────────────────────────────────────────
FRAME_COLUMNS = [
    "timestamp_unix", "timestamp_monotonic", "elapsed_s", "frame_idx",
    "state", "tags_detected", "tags_used_after_gating", "tags_in_graph",
    "qualified_tags", "single_tag_frames_rejected_total",
    "median_residual_px", "worst_residual_px",
    "last_jump_mm", "last_jump_residual_mm",
    "camera_x_m", "camera_y_m", "camera_z_m",
    "camera_qw", "camera_qx", "camera_qy", "camera_qz",
    "anchor_observed_this_frame",
    "gantry_x_mm", "gantry_y_mm", "gantry_z_mm",
    "gantry_vx_mm_s", "gantry_vy_mm_s", "gantry_vz_mm_s",
    "in_warmup", "backend_update_ms",
]
TAG_HISTORY_COLUMNS = [
    "timestamp_unix", "timestamp_monotonic", "elapsed_s", "frame_idx",
    "tag_id", "event_type",
    "position_x_m", "position_y_m", "position_z_m",
    "quaternion_wxyz_w", "quaternion_wxyz_x", "quaternion_wxyz_y", "quaternion_wxyz_z",
    "n_observations_so_far", "uncertainty_mm", "shift_mm_since_last_event",
]
BATCH_COLUMNS = [
    "timestamp_unix", "timestamp_monotonic", "elapsed_s", "frame_idx",
    "trigger_reason", "n_tags_in_graph", "n_camera_poses_in_graph",
    "batch_iterations", "batch_initial_error", "batch_final_error",
    "max_tag_shift_mm", "median_tag_shift_mm", "isam2_rebuilt",
    "batch_wallclock_ms", "anchor_drifted_mm",
]
USER_ACTION_COLUMNS = [
    "timestamp_unix", "timestamp_monotonic", "elapsed_s",
    "action_type", "detail", "gantry_x_mm", "gantry_y_mm", "gantry_z_mm",
]
ANCHOR_COLUMNS = [
    "timestamp_unix", "timestamp_monotonic", "elapsed_s",
    "anchor_tag_id",
    "anchor_position_x_m", "anchor_position_y_m", "anchor_position_z_m",
    "anchor_position_drift_from_t0_mm", "anchor_rotation_drift_from_t0_deg",
    "anchor_uncertainty_mm",
]
LOOP_COLUMNS = [
    "timestamp_unix", "timestamp_monotonic", "elapsed_s", "frame_idx",
    "revisit_target_t_s", "distance_m", "n_tags_co_observed",
]
SLAM_COLUMNS = [
    "timestamp_unix", "timestamp_monotonic", "elapsed_s", "frame_idx",
    "n_variables_in_isam", "n_factors_in_isam",
    "n_relinearizations_since_last_sample",
    "isam_update_ms_p50", "isam_update_ms_p95",
    "relinearize_threshold_active", "periodic_batch_due_in_s",
]
USER_NOTE_COLUMNS = [
    "timestamp_unix", "timestamp_monotonic", "elapsed_s", "frame_idx",
    "note_text", "camera_x_m", "camera_y_m", "camera_z_m",
    "gantry_x_mm", "gantry_y_mm", "gantry_z_mm",
]

_FILES = {
    "frame": ("survey_diagnostics.csv", FRAME_COLUMNS),
    "tag": ("tag_history.csv", TAG_HISTORY_COLUMNS),
    "batch": ("batch_events.csv", BATCH_COLUMNS),
    "action": ("user_actions.csv", USER_ACTION_COLUMNS),
    "anchor": ("anchor_stability.csv", ANCHOR_COLUMNS),
    "loop": ("loop_closure_events.csv", LOOP_COLUMNS),
    "slam": ("slam_internals.csv", SLAM_COLUMNS),
    "note": ("user_notes.csv", USER_NOTE_COLUMNS),
}

_DRIFT_ONSET_BASELINE_MAX = 200      # frames
_DRIFT_ONSET_SUSTAIN = 5             # consecutive tripped frames


def _f(v) -> str:
    """CSV cell formatter: NaN/None -> '', float -> 6 sig, bool -> True/False."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.6f}"
    return str(v)


def _yaml_float(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "null"
    return f"{float(v):.6f}"


class SurveyDiagnosticsRecorder:
    def __init__(self, run_dir: Path, t0_monotonic: float, *,
                 flush_interval_s: float = 1.0,
                 relinearize_threshold: float = 0.001) -> None:
        self._dir = Path(run_dir)
        self._t0 = float(t0_monotonic)
        self._flush_interval = float(flush_interval_s)
        self._relin_threshold = float(relinearize_threshold)
        self._snap_dir = self._dir / "tag_snapshots"

        self._q: Queue = Queue()
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._fhs: dict[str, object] = {}
        self._writers: dict[str, object] = {}

        # aggregates (touched ONLY on the daemon thread)
        self._agg = {
            "n_frames": 0,
            "residuals": [],            # finite median_residual_px per frame
            "frame_series": [],         # (frame_idx, elapsed, med_res, jump_res, warmup)
            "n_batch_events": 0,
            "n_isam_rebuilds": 0,
            "max_batch_shift_mm": 0.0,
            "frame_of_max_shift": None,
            "max_anchor_drift_mm": 0.0,
            "anchor_drift_at_end": 0.0,
            "n_loop": 0,
            "n_excluded": 0,
            "n_notes": 0,
            "n_warnings": 0,
            "duplicate_ids": set(),     # tag IDs auto-flagged as physical duplicates
            "tags": {},                 # tid -> {first_seen_t_s, promoted_t_s, n_obs, final_unc_mm}
        }
        self._summary_meta: dict | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._snap_dir.mkdir(parents=True, exist_ok=True)
        for key, (name, cols) in _FILES.items():
            fh = open(self._dir / name, "w", newline="", encoding="utf-8")
            wr = csv.writer(fh)
            wr.writerow(cols)
            fh.flush()
            self._fhs[key] = fh
            self._writers[key] = wr
        self._stop.clear()
        self._closed.clear()
        self._thread = threading.Thread(target=self._run, name="survey-diag",
                                        daemon=True)
        self._thread.start()

    def close(self, summary_meta: dict | None = None, timeout: float = 5.0) -> None:
        """Drain the queue, write diagnostics_summary.json, close files."""
        if self._thread is None:
            return
        self._summary_meta = dict(summary_meta or {})
        self._stop.set()
        self._q.put(("__close__", None))
        self._closed.wait(timeout=timeout)
        self._thread = None

    # ── producer-side log methods (cheap; build payload + enqueue) ───────────
    def _stamp(self, elapsed_s: float | None = None, t_mono: float | None = None):
        if t_mono is not None:
            mono = float(t_mono)
            elapsed = mono - self._t0
        else:
            elapsed = float(elapsed_s if elapsed_s is not None else 0.0)
            mono = self._t0 + elapsed
        return time.time(), mono, elapsed

    def log_frame(self, d: dict) -> None:
        ts, mono, el = self._stamp(elapsed_s=d.get("elapsed_s"))
        self._q.put(("frame", {"ts": ts, "mono": mono, "el": el, **d}))

    def log_tag_event(self, event_type: str, tag_id: int, t_mono: float,
                      frame_idx: int, *, pose=None, n_obs: int = 0,
                      shift_mm=None, uncertainty_mm=None) -> None:
        ts, mono, el = self._stamp(t_mono=t_mono)
        pos = qwxyz = None
        if pose is not None:
            try:
                t = pose.translation()
                pos = (float(t[0]), float(t[1]), float(t[2]))
                qwxyz = _quat_wxyz(pose.rotation())
            except Exception:
                pos = qwxyz = None
        self._q.put(("tag", {
            "ts": ts, "mono": mono, "el": el, "frame_idx": int(frame_idx),
            "tag_id": int(tag_id), "event_type": str(event_type),
            "pos": pos, "qwxyz": qwxyz, "n_obs": int(n_obs),
            "unc": uncertainty_mm, "shift": shift_mm,
        }))

    def log_batch_event(self, ev: dict) -> None:
        ts, mono, el = self._stamp(elapsed_s=ev.get("elapsed_s"))
        self._q.put(("batch", {"ts": ts, "mono": mono, "el": el, **ev}))

    def log_user_action(self, action_type: str, detail: dict | None,
                        gantry_mm) -> None:
        ts, mono, el = self._stamp(t_mono=time.monotonic())
        self._q.put(("action", {
            "ts": ts, "mono": mono, "el": el,
            "action_type": str(action_type),
            "detail": json.dumps(detail or {}, separators=(",", ":")),
            "gantry": _xyz(gantry_mm),
        }))

    def log_anchor_stability(self, elapsed_s: float, anchor_id: int, pos_m,
                             drift_mm: float, rot_deg: float, unc_mm: float) -> None:
        ts, mono, el = self._stamp(elapsed_s=elapsed_s)
        self._q.put(("anchor", {
            "ts": ts, "mono": mono, "el": el, "anchor_id": int(anchor_id),
            "pos": _xyz(pos_m), "drift_mm": float(drift_mm),
            "rot_deg": float(rot_deg), "unc_mm": unc_mm,
        }))

    def log_loop_closure(self, elapsed_s: float, frame_idx: int, *,
                         revisit_target_t_s: float, distance_m: float,
                         n_tags_co_observed: int) -> None:
        ts, mono, el = self._stamp(elapsed_s=elapsed_s)
        self._q.put(("loop", {
            "ts": ts, "mono": mono, "el": el, "frame_idx": int(frame_idx),
            "target_t": float(revisit_target_t_s), "dist": float(distance_m),
            "co": int(n_tags_co_observed),
        }))

    def log_slam_internals(self, d: dict) -> None:
        ts, mono, el = self._stamp(elapsed_s=d.get("elapsed_s"))
        self._q.put(("slam", {"ts": ts, "mono": mono, "el": el, **d}))

    def log_user_note(self, note_text: str, frame_idx: int, camera_xyz,
                      gantry_mm) -> None:
        ts, mono, el = self._stamp(t_mono=time.monotonic())
        self._q.put(("note", {
            "ts": ts, "mono": mono, "el": el, "frame_idx": int(frame_idx),
            "note": str(note_text), "cam": _xyz(camera_xyz),
            "gantry": _xyz(gantry_mm),
        }))

    def log_snapshot(self, elapsed_s: float, frame_idx: int, anchor_id,
                     tag_states: dict, batch_recent: bool) -> None:
        self._q.put(("snapshot", {
            "el": float(elapsed_s), "frame_idx": int(frame_idx),
            "anchor_id": anchor_id, "tags": dict(tag_states),
            "batch_recent": bool(batch_recent),
        }))

    def log_warning(self, text: str) -> None:
        self._q.put(("warning", {"text": str(text)}))

    def set_final_tag_stats(self, tags: dict) -> None:
        """Merge batch-optimized per-tag uncertainty / obs counts into the summary
        (call right before close() so tag_summary.final_unc_mm reflects the batch)."""
        self._q.put(("finalstats", dict(tags)))

    # ── daemon side ───────────────────────────────────────────────────────────
    def _run(self) -> None:
        last_flush = time.monotonic()
        while True:
            try:
                item = self._q.get(timeout=0.25)
            except Empty:
                item = None
            if item is not None:
                kind, payload = item
                if kind == "__close__":
                    self._drain_remaining()
                    self._write_summary()
                    self._flush_all()
                    break
                try:
                    self._handle(kind, payload)
                except Exception as exc:  # never let logging crash the daemon
                    print(f"[survey-diag] write error ({kind}): {exc}",
                          file=sys.stderr)
            now = time.monotonic()
            if (now - last_flush) >= self._flush_interval:
                self._flush_all()
                last_flush = now
        self._close_files()
        self._closed.set()

    def _drain_remaining(self) -> None:
        while True:
            try:
                kind, payload = self._q.get_nowait()
            except Empty:
                return
            if kind == "__close__":
                continue
            try:
                self._handle(kind, payload)
            except Exception:
                pass

    def _handle(self, kind: str, p: dict) -> None:
        if kind == "frame":
            self._writers["frame"].writerow(self._row_frame(p))
            self._agg["n_frames"] += 1
            mr = p.get("median_residual_px")
            if isinstance(mr, float) and math.isfinite(mr):
                self._agg["residuals"].append(mr)
            self._agg["frame_series"].append((
                p.get("frame_idx"), p["el"],
                _num(p.get("median_residual_px")),
                _num(p.get("last_jump_residual_mm")),
                bool(p.get("in_warmup")),
            ))
        elif kind == "tag":
            self._writers["tag"].writerow(self._row_tag(p))
            self._agg_tag(p)
        elif kind == "batch":
            self._writers["batch"].writerow(self._row_batch(p))
            self._agg["n_batch_events"] += 1
            if p.get("isam2_rebuilt"):
                self._agg["n_isam_rebuilds"] += 1
            ms = float(p.get("max_tag_shift_mm", 0.0) or 0.0)
            if ms > self._agg["max_batch_shift_mm"]:
                self._agg["max_batch_shift_mm"] = ms
                self._agg["frame_of_max_shift"] = p.get("frame_idx")
        elif kind == "action":
            self._writers["action"].writerow(self._row_action(p))
        elif kind == "anchor":
            self._writers["anchor"].writerow(self._row_anchor(p))
            d = float(p.get("drift_mm", 0.0) or 0.0)
            self._agg["max_anchor_drift_mm"] = max(self._agg["max_anchor_drift_mm"], d)
            self._agg["anchor_drift_at_end"] = d
        elif kind == "loop":
            self._writers["loop"].writerow(self._row_loop(p))
            self._agg["n_loop"] += 1
        elif kind == "slam":
            self._writers["slam"].writerow(self._row_slam(p))
        elif kind == "note":
            self._writers["note"].writerow(self._row_note(p))
            self._agg["n_notes"] += 1
        elif kind == "snapshot":
            self._write_snapshot(p)
        elif kind == "warning":
            self._agg["n_warnings"] += 1
        elif kind == "finalstats":
            for tid, s in p.items():
                rec = self._agg["tags"].setdefault(
                    int(tid), {"first_seen_t_s": None, "promoted_t_s": None,
                               "n_obs": 0, "final_unc_mm": None})
                unc = s.get("uncertainty_mm")
                if isinstance(unc, (int, float)) and math.isfinite(float(unc)):
                    rec["final_unc_mm"] = round(float(unc), 3)
                rec["n_obs"] = max(rec["n_obs"], int(s.get("n_observations", 0)))

    # — row builders —
    def _row_frame(self, p: dict) -> list:
        cam = p.get("camera", [None] * 3)
        q = p.get("quat", [None] * 4)
        g = p.get("gantry_mm", [None] * 3)
        gv = p.get("gantry_vel_mm_s", [None] * 3)
        return [_f(p["ts"]), _f(p["mono"]), _f(p["el"]), _f(p.get("frame_idx")),
                _f(p.get("state")), _f(p.get("tags_detected")),
                _f(p.get("tags_used_after_gating")), _f(p.get("tags_in_graph")),
                _f(p.get("qualified_tags")),
                _f(p.get("single_tag_frames_rejected_total")),
                _f(p.get("median_residual_px")), _f(p.get("worst_residual_px")),
                _f(p.get("last_jump_mm")), _f(p.get("last_jump_residual_mm")),
                _f(cam[0]), _f(cam[1]), _f(cam[2]),
                _f(q[0]), _f(q[1]), _f(q[2]), _f(q[3]),
                _f(bool(p.get("anchor_observed_this_frame"))),
                _f(g[0]), _f(g[1]), _f(g[2]),
                _f(gv[0]), _f(gv[1]), _f(gv[2]),
                _f(bool(p.get("in_warmup"))), _f(p.get("backend_update_ms"))]

    def _row_tag(self, p: dict) -> list:
        pos = p.get("pos") or (None, None, None)
        q = p.get("qwxyz") or (None, None, None, None)
        return [_f(p["ts"]), _f(p["mono"]), _f(p["el"]), _f(p["frame_idx"]),
                _f(p["tag_id"]), _f(p["event_type"]),
                _f(pos[0]), _f(pos[1]), _f(pos[2]),
                _f(q[0]), _f(q[1]), _f(q[2]), _f(q[3]),
                _f(p["n_obs"]),
                _f(p["unc"] if p["unc"] is not None else float("nan")),
                _f(p["shift"] if p["shift"] is not None else float("nan"))]

    def _row_batch(self, p: dict) -> list:
        return [_f(p["ts"]), _f(p["mono"]), _f(p["el"]), _f(p.get("frame_idx")),
                _f(p.get("trigger_reason")), _f(p.get("n_tags_in_graph")),
                _f(p.get("n_camera_poses_in_graph")), _f(p.get("batch_iterations")),
                _f(p.get("batch_initial_error")), _f(p.get("batch_final_error")),
                _f(p.get("max_tag_shift_mm")), _f(p.get("median_tag_shift_mm")),
                _f(bool(p.get("isam2_rebuilt"))), _f(p.get("batch_wallclock_ms")),
                _f(p.get("anchor_drifted_mm"))]

    def _row_action(self, p: dict) -> list:
        g = p.get("gantry", [None] * 3)
        return [_f(p["ts"]), _f(p["mono"]), _f(p["el"]),
                _f(p["action_type"]), p["detail"], _f(g[0]), _f(g[1]), _f(g[2])]

    def _row_anchor(self, p: dict) -> list:
        pos = p.get("pos", [None] * 3)
        return [_f(p["ts"]), _f(p["mono"]), _f(p["el"]), _f(p["anchor_id"]),
                _f(pos[0]), _f(pos[1]), _f(pos[2]),
                _f(p["drift_mm"]), _f(p["rot_deg"]),
                _f(p["unc_mm"] if p["unc_mm"] is not None else float("nan"))]

    def _row_loop(self, p: dict) -> list:
        return [_f(p["ts"]), _f(p["mono"]), _f(p["el"]), _f(p["frame_idx"]),
                _f(p["target_t"]), _f(p["dist"]), _f(p["co"])]

    def _row_slam(self, p: dict) -> list:
        return [_f(p["ts"]), _f(p["mono"]), _f(p["el"]), _f(p.get("frame_idx")),
                _f(p.get("n_variables")), _f(p.get("n_factors")),
                _f(p.get("n_relin")), _f(p.get("p50")), _f(p.get("p95")),
                _f(p.get("relin_threshold")), _f(p.get("batch_due_in_s"))]

    def _row_note(self, p: dict) -> list:
        c = p.get("cam", [None] * 3)
        g = p.get("gantry", [None] * 3)
        return [_f(p["ts"]), _f(p["mono"]), _f(p["el"]), _f(p["frame_idx"]),
                p["note"], _f(c[0]), _f(c[1]), _f(c[2]),
                _f(g[0]), _f(g[1]), _f(g[2])]

    def _agg_tag(self, p: dict) -> None:
        tid = p["tag_id"]
        rec = self._agg["tags"].setdefault(
            tid, {"first_seen_t_s": None, "promoted_t_s": None,
                  "n_obs": 0, "final_unc_mm": None})
        ev = p["event_type"]
        if ev == "first_seen" and rec["first_seen_t_s"] is None:
            rec["first_seen_t_s"] = round(p["el"], 3)
        if ev == "promoted" and rec["promoted_t_s"] is None:
            rec["promoted_t_s"] = round(p["el"], 3)
        if ev == "excluded":
            self._agg["n_excluded"] += 1
        if ev == "duplicate_detected":
            self._agg["duplicate_ids"].add(int(tid))
        if p["n_obs"]:
            rec["n_obs"] = max(rec["n_obs"], int(p["n_obs"]))
        if isinstance(p["unc"], (int, float)) and p["unc"] is not None \
                and math.isfinite(float(p["unc"])):
            rec["final_unc_mm"] = round(float(p["unc"]), 3)

    # — snapshots —
    def _write_snapshot(self, p: dict) -> None:
        el = p["el"]
        path = self._snap_dir / f"snapshot_t{int(round(el)):03d}s.yaml"
        lines = [
            f"elapsed_s: {el:.1f}",
            f"frame_idx: {p['frame_idx']}",
            f"anchor_tag_id: {p['anchor_id'] if p['anchor_id'] is not None else 'null'}",
            f"batch_event_at_this_snapshot: {'true' if p['batch_recent'] else 'false'}",
            "tags:",
        ]
        for tid in sorted(p["tags"]):
            s = p["tags"][tid]
            pos = s.get("position_m", [0.0, 0.0, 0.0])
            q = s.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
            lines.append(f"  {int(tid)}:")
            lines.append(f"    position_m: [{_yaml_float(pos[0])}, "
                         f"{_yaml_float(pos[1])}, {_yaml_float(pos[2])}]")
            lines.append(f"    quaternion_wxyz: [{_yaml_float(q[0])}, "
                         f"{_yaml_float(q[1])}, {_yaml_float(q[2])}, {_yaml_float(q[3])}]")
            lines.append(f"    n_observations: {int(s.get('n_observations', 0))}")
            lines.append(f"    uncertainty_mm: {_yaml_float(s.get('uncertainty_mm'))}")
        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            print(f"[survey-diag] snapshot write failed: {exc}", file=sys.stderr)

    # — summary —
    def _drift_onset(self):
        post = [s for s in self._agg["frame_series"] if not s[4]]  # after warmup
        if len(post) < 20:
            return (None, None)
        base_n = min(_DRIFT_ONSET_BASELINE_MAX, max(10, int(0.2 * len(post))))
        base = post[:base_n]

        def stats(vals):
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if not vals:
                return (0.0, 0.0)
            mu = statistics.fmean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            return (mu, sd)

        mu_r, sd_r = stats([s[2] for s in base])
        mu_j, sd_j = stats([s[3] for s in base])
        thr_r, thr_j = mu_r + 3 * sd_r, mu_j + 3 * sd_j
        run, run_start = 0, None
        for s in post[base_n:]:
            r, j = s[2], s[3]
            trip = ((r is not None and math.isfinite(r) and r > thr_r) or
                    (j is not None and math.isfinite(j) and j > thr_j))
            if trip:
                if run_start is None:
                    run_start = s
                run += 1
                if run >= _DRIFT_ONSET_SUSTAIN:
                    return (run_start[0], round(run_start[1], 3))
            else:
                run, run_start = 0, None
        return (None, None)

    def _write_summary(self) -> None:
        meta = self._summary_meta or {}
        res = self._agg["residuals"]
        onset_frame, onset_t = self._drift_onset()
        summary = {
            "run_metadata": {
                "anchor_tag_id": meta.get("anchor_tag_id"),
                "survey_duration_s": meta.get("survey_duration_s"),
                "n_frames_processed": meta.get("n_frames_processed",
                                               self._agg["n_frames"]),
                "n_tags_qualified": meta.get("n_tags_qualified"),
                "fisheye_calib_path": meta.get("fisheye_calib_path"),
                "constants": meta.get("constants", {}),
                "git_commit": meta.get("git_commit"),
                "tool_version": meta.get("tool_version"),
            },
            "summary_stats": {
                "median_residual_px_avg": round(statistics.fmean(res), 4) if res else None,
                "median_residual_px_p95": round(_percentile(res, 95), 4) if res else None,
                "n_batch_events": self._agg["n_batch_events"],
                "n_isam_rebuilds": self._agg["n_isam_rebuilds"],
                "max_anchor_drift_mm": round(self._agg["max_anchor_drift_mm"], 3),
                "loop_closure_events": self._agg["n_loop"],
                "tags_excluded_by_user": self._agg["n_excluded"],
                "duplicate_ids_detected": sorted(self._agg["duplicate_ids"]),
                "user_notes_count": self._agg["n_notes"],
                "warnings_logged": self._agg["n_warnings"],
            },
            "tag_summary": {
                str(tid): rec for tid, rec in sorted(self._agg["tags"].items())
            },
            "drift_diagnostics": {
                "anchor_drift_mm_at_end": round(self._agg["anchor_drift_at_end"], 3),
                "max_periodic_batch_shift_mm": round(self._agg["max_batch_shift_mm"], 3),
                "frame_idx_of_max_shift": self._agg["frame_of_max_shift"],
                "approximate_drift_onset_frame": onset_frame,
                "approximate_drift_onset_t_s": onset_t,
            },
        }
        try:
            (self._dir / "diagnostics_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            print(f"[survey-diag] summary write failed: {exc}", file=sys.stderr)

    # — file io —
    def _flush_all(self) -> None:
        for fh in self._fhs.values():
            try:
                fh.flush()
            except Exception:
                pass

    def _close_files(self) -> None:
        for fh in self._fhs.values():
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
        self._fhs.clear()
        self._writers.clear()


def _quat_wxyz(rot):
    """Quaternion [w, x, y, z] across GTSAM wrapper versions (no gtsam import)."""
    try:
        q = rot.toQuaternion()
        return (float(q.w()), float(q.x()), float(q.y()), float(q.z()))
    except Exception:
        q = rot.quaternion()
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _xyz(v):
    if v is None:
        return [None, None, None]
    try:
        return [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        return [None, None, None]


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _percentile(vals, pct):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
