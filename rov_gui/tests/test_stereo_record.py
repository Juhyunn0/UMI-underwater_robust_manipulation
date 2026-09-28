#!/usr/bin/env python3
"""
test_stereo_record.py — the raw mono-pair recorder (``--record-stereo``).

The plumbing and the five safety rules are shared with the depth recorder and
are pinned by ``test_depth_record.py``; these tests cover what is NEW here: a
pair is written as a pair or not at all, the interval sampler, the metadata
that makes a recording regenerable (the rig — without it the PNGs are pixels
with no geometry), the CLI refusals, and the wiring that files the pairs in the
controller's run folder.

    python -m pytest rov_gui/tests/test_stereo_record.py -q
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rov_gui.perception.stereo_record import (DEFAULT_MAX_PAIRS,  # noqa: E402
                                              StereoRecorder)


def _mono(i: int = 0, w: int = 64, h: int = 40):
    return np.full((h, w), i % 256, np.uint8)


def _drain(rec, tries: int = 60):
    for _ in range(tries):
        if rec.counters["written"] + rec.counters["write_errors_writer"] \
                >= rec.counters["offered"] - rec.counters["dropped_full"] \
                - rec.counters["dropped_budget"] - rec.counters["dropped_no_writer"] \
                - rec.counters["dropped_interval"]:
            return
        time.sleep(0.02)


@pytest.fixture()
def run_dir():
    with tempfile.TemporaryDirectory(prefix="rovgui_stereo_") as d:
        yield Path(d)


# ---------------------------------------------------------------- happy path
def test_pairs_are_written_and_read_back(run_dir):
    rec = StereoRecorder(lambda: run_dir)
    assert rec.start(rig_describe={"fx_rect": 249.5}, fstereo={"iters": 8})
    for i in range(5):
        rec.submit(100.0 + i / 7.26, _mono(i), _mono(i + 100), frame_seq=i)
        _drain(rec)                    # a feed rate the disk can keep up with
    c = rec.close()

    assert c["written"] == 5 and c["dropped_full"] == 0
    d = rec.dir
    assert len(list((d / "left").glob("*.png"))) == 5
    assert len(list((d / "right").glob("*.png"))) == 5

    rows = (d / "index.csv").read_text().strip().splitlines()
    assert rows[0] == "seq,t_capture,frame_seq,left_png,right_png"
    first = rows[1].split(",")
    assert first[0] == "0" and first[2] == "0"
    assert first[3] == "left/000000.png" and first[4] == "right/000000.png"

    # Lossless: the bytes that come back ARE the bytes that went in. A lossy
    # codec here would make a regenerated disparity a function of the codec.
    import cv2
    back = cv2.imread(str(d / "left/000003.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(back, _mono(3))


def test_meta_carries_the_rig_and_the_settings_that_flew(run_dir):
    """Without the rig the PNGs are pixels with no geometry, and without the
    flown settings a re-run cannot say what it is being compared against."""
    rig = {"fx_rect": 249.5656, "baseline_mm": 75.3, "k_mm_px": 18800.0,
           "provenance": {"source": "EEPROM"}}
    rec = StereoRecorder(lambda: run_dir)
    rec.start(rig_describe=rig, fstereo={"iters": 8, "scale": 0.75,
                                         "ckpt": "23-51-11/model_best_bp2.pth"})
    rec.submit(1.0, _mono(1), _mono(2), frame_seq=7)
    _drain(rec)
    rec.close()
    meta = json.loads((rec.dir / "meta.json").read_text())
    assert meta["schema"] == "rov_gui/stereo_pair_recording/1"
    assert meta["rig"]["fx_rect"] == 249.5656
    assert meta["fstereo"]["iters"] == 8
    assert meta["mono_shape"] == [40, 64] and meta["mono_dtype"] == "uint8"
    assert "RAW" in meta["images"] and "NOT" in meta["images"]
    assert meta["counters"]["written"] == 1
    assert meta["counts_complete"] is True


# ------------------------------------------------------------------ sampling
def test_every_s_samples_on_capture_time_and_counts_what_it_skipped(run_dir):
    """A sparse recording must be visibly sparse, not look like a starved feed."""
    rec = StereoRecorder(lambda: run_dir, every_s=1.0)
    rec.start()
    for i in range(21):                      # 0.0 .. 2.0 s at 10 Hz
        rec.submit(i * 0.1, _mono(i), _mono(i), frame_seq=i)
    _drain(rec)
    c = rec.close()
    assert c["written"] == 3, "t = 0.0, 1.0, 2.0"
    assert c["dropped_interval"] == 18
    assert c["offered"] == 21, "a skipped pair is still an offer"


def test_a_stalled_then_resumed_feed_does_not_burst(run_dir):
    """The gate is against the last KEPT stamp, not an accumulating deadline:
    after a 10 s gap exactly one pair is kept, not ten."""
    rec = StereoRecorder(lambda: run_dir, every_s=1.0)
    rec.start()
    rec.submit(0.0, _mono(0), _mono(0))
    for k in range(5):                       # all inside one interval, at t+10
        rec.submit(10.0 + k * 0.05, _mono(k), _mono(k))
    _drain(rec)
    c = rec.close()
    assert c["written"] == 2


def test_every_zero_keeps_every_pair(run_dir):
    rec = StereoRecorder(lambda: run_dir, every_s=0.0)
    rec.start()
    for i in range(6):
        rec.submit(i * 0.001, _mono(i), _mono(i))
        _drain(rec)                    # feed it at a rate the disk can take
    c = rec.close()
    # Not "written == 6": QUEUE_DEPTH is 4 and a loaded machine may still drop
    # one for a full queue. The claim under test is that NONE was dropped by
    # the interval gate, which is off.
    assert c["dropped_interval"] == 0
    assert c["written"] + c["dropped_full"] == 6


# ---------------------------------------------------------------- robustness
def test_half_a_pair_is_never_recorded(run_dir):
    """If the right image cannot be written the row is not written either.

    A row naming a file that does not exist is worse than no row: an offline
    pass would read it, fail on the missing partner, and blame the camera.
    """
    rec = StereoRecorder(lambda: run_dir)
    rec.start()
    rec.submit(0.0, _mono(0), _mono(0))
    _drain(rec)
    os.chmod(rec.dir / "right", stat.S_IREAD | stat.S_IEXEC)
    try:
        for i in range(1, 5):
            rec.submit(float(i), _mono(i), _mono(i))
        _drain(rec)
        c = rec.close()
    finally:
        os.chmod(rec.dir / "right", stat.S_IRWXU)
    rows = len((rec.dir / "index.csv").read_text().strip().splitlines()) - 1
    right_on_disk = len(list((rec.dir / "right").glob("*.png")))
    assert c["written"] == rows == right_on_disk == 1
    assert c["write_errors_writer"] > 0


def test_budget_stops_writing(run_dir):
    rec = StereoRecorder(lambda: run_dir, max_frames=2)
    rec.start()
    for i in range(6):
        rec.submit(float(i), _mono(i), _mono(i))
    _drain(rec)
    c = rec.close()
    assert c["written"] == 2 and c["dropped_budget"] == 4


def test_submit_never_blocks(run_dir):
    """The producer is the learned-depth tick, already carrying a ~46 ms
    inference. A full queue must drop, not wait."""
    rec = StereoRecorder(lambda: run_dir)
    rec.start()
    t0 = time.monotonic()
    for i in range(300):
        rec.submit(float(i), _mono(i), _mono(i))
    dt = time.monotonic() - t0
    offered = rec.counters["offered"]
    rec.close()
    assert offered == 300
    assert rec.counters["dropped_full"] > 0
    assert dt / 300 < 1.0e-3, f"{1e3 * dt / 300:.2f} ms per submit — too slow"


def test_no_folder_before_the_first_pair(run_dir):
    rec = StereoRecorder(lambda: run_dir)
    rec.start()
    time.sleep(0.05)
    assert not (run_dir / "stereo").exists(), \
        "the run folder must be created by the writer, on the first pair"
    assert rec.dir is None
    rec.submit(0.0, _mono(), _mono())
    _drain(rec)
    assert (run_dir / "stereo" / "index.csv").exists()
    rec.close()


def test_unwritable_run_folder_disables_rather_than_raises(run_dir):
    ro = run_dir / "ro"
    ro.mkdir()
    os.chmod(ro, stat.S_IREAD | stat.S_IEXEC)
    try:
        rec = StereoRecorder(lambda: ro)
        rec.start()
        rec.submit(0.0, _mono(), _mono())     # must not raise
        time.sleep(0.1)
        rec.submit(1.0, _mono(), _mono())
        c = rec.close()
        assert c["written"] == 0
        assert rec.dir is None and rec.why
    finally:
        os.chmod(ro, stat.S_IRWXU)


def test_describe_is_safe_at_every_stage(run_dir):
    rec = StereoRecorder(lambda: run_dir, every_s=0.5)
    d = rec.describe()
    assert set(d) >= {"enabled", "started", "dir", "why", "counters", "every_s"}
    assert d["started"] is False and d["why"]
    rec.start()
    rec.submit(0.0, _mono(), _mono())
    _drain(rec)
    rec.close()
    d = rec.describe()
    assert d["every_s"] == 0.5 and d["mono_shape"] == (40, 64)
    assert isinstance(d["counters"], dict)


def test_close_is_idempotent_and_submit_after_close_is_a_noop(run_dir):
    rec = StereoRecorder(lambda: run_dir)
    rec.start()
    rec.submit(0.0, _mono(), _mono())
    _drain(rec)
    a = rec.close()
    b = rec.close()
    rec.submit(1.0, _mono(), _mono())
    assert a["written"] == b["written"] == 1
    assert rec.counters["dropped_disabled"] >= 1


# ------------------------------------------------------------------ the flag
def _args(argv):
    from rov_gui.__main__ import build_parser
    return build_parser().parse_args(argv)


def test_record_stereo_without_fstereo_is_refused():
    from rov_gui.__main__ import check_policy
    r = check_policy(_args(["--source", "hw", "--mpc", "--record-stereo"]))
    assert r is not None and r[0] == "refuse" and "--fstereo" in r[1]


def test_record_stereo_without_mpc_is_refused():
    from rov_gui.__main__ import check_policy
    r = check_policy(_args(["--source", "hw", "--fstereo", "--record-stereo"]))
    assert r is not None and r[0] == "refuse" and "--mpc" in r[1]


def test_record_stereo_budgets_must_be_sane():
    from rov_gui.__main__ import check_policy
    base = ["--source", "hw", "--mpc", "--fstereo", "--record-stereo"]
    assert check_policy(_args(base)) is None
    assert check_policy(_args(base + ["--record-stereo-max", "0"]))[0] == "refuse"
    assert check_policy(_args(base + ["--record-stereo-every", "-1"]))[0] == "refuse"
    assert check_policy(_args(base + ["--record-stereo-every", "2"])) is None


def test_the_pairs_are_filed_in_the_controllers_run_folder():
    """Same property --record-depth has, and for the same reason: MpcWorker
    owns the run TREE and the PIN, so a recorder that resolved its own folder
    could file a dive's pairs away from the CSVs they belong to."""
    import inspect
    from rov_gui.backends import hardware as H
    src = inspect.getsource(H.FStereoWorker.setup)
    assert "StereoRecorder(\n                    self.run_dir_fn," in src \
        or "StereoRecorder(self.run_dir_fn" in src, \
        "the recorder must be given the controller's run folder"
    assert "self.fstereo.run_dir_fn = self.mpc._run_dir" in \
        inspect.getsource(H.HardwareBackend.__init__)


def test_the_run_record_never_claims_the_flag_was_not_given():
    """A construction failure must not make meta() imply --record-stereo was
    never asked for — the silent no-op the refusals exist to prevent."""
    import inspect
    from rov_gui.backends import hardware as H
    assert "_rec_why" in inspect.getsource(H.FStereoWorker.setup)
    assert "stereo_recording" in inspect.getsource(H.FStereoWorker.meta)


def test_a_recorder_defect_never_stops_learned_depth():
    """Rule 3 at the call site: _record_pair swallows everything and disables
    itself, because a diagnostic must not cost the run its depth."""
    import inspect
    from rov_gui.backends import hardware as H
    src = inspect.getsource(H.FStereoWorker._record_pair)
    assert "except Exception" in src and "self.stereo_rec = None" in src
    tick = inspect.getsource(H.FStereoWorker.tick)
    assert "self._record_pair(item)" in tick
    assert tick.index("self._record_pair(item)") < tick.index("s.infer("), \
        "the pair must be recorded BEFORE inference can throw it away"


def test_default_budget_is_documented_and_positive():
    assert DEFAULT_MAX_PAIRS > 0
    from rov_gui.perception import stereo_record as SR
    assert "296 KiB" in SR.__doc__, "the size measurement must stay stated"


def main() -> int:
    """Runner, so `python rov_gui/tests/test_stereo_record.py` works too."""
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        needs_dir = "run_dir" in fn.__code__.co_varnames[:fn.__code__.co_argcount]
        try:
            if needs_dir:
                with tempfile.TemporaryDirectory(prefix="rovgui_stereo_") as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  ok    {name}")
        except Exception:                                        # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
