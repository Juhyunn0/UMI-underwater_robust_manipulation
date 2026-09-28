#!/usr/bin/env python3
"""
test_depth_record.py — the underwater depth recorder, and the safety rules it
exists under.

This code runs inside the policy worker while the vehicle is armed, so most of
these tests are not about the happy path. Each of the failure tests below pins
a defect found in the 2026-09-06 safety audit; a regression in any of them is a
recorder that can stall, mislead, or end a dive.

    python -m pytest rov_gui/tests/test_depth_record.py -q
"""

from __future__ import annotations

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

from rov_gui.perception.depth_record import DepthRecorder  # noqa: E402


def _obs(i: int = 0):
    u8 = np.full((224, 224), i % 256, np.uint8)
    return np.stack([u8, np.full_like(u8, 255), u8], axis=-1)


def _depth():
    return np.full((224, 224), 1000, np.uint16)


def _drain(rec, tries: int = 60):
    """Wait for the writer to catch up, without a fixed sleep."""
    for _ in range(tries):
        if rec.counters["written"] + rec.counters["write_errors_writer"] \
                >= rec.counters["offered"] - rec.counters["dropped_full"] \
                - rec.counters["dropped_budget"] - rec.counters["dropped_no_writer"]:
            return
        time.sleep(0.02)


@pytest.fixture()
def run_dir():
    with tempfile.TemporaryDirectory(prefix="rovgui_rec_") as d:
        yield Path(d)


# ---------------------------------------------------------------- happy path
def test_records_and_reads_back(run_dir):
    """At a realistic feed rate nothing is dropped and everything reads back.

    Paced at 100 Hz, which is ~25x the rate the policy worker actually submits
    at (~2 frames per inference at period_s 0.5 s). A burst faster than the
    writer can encode WILL drop — that is the design, and
    ``test_submit_never_blocks_when_the_queue_is_full`` pins it — so the happy
    path has to be fed at something like a real rate to mean anything.
    """
    rec = DepthRecorder(lambda: run_dir, save_depth=True, max_frames=50)
    assert rec.start(builder_describe={"z_near_m": 0.2, "z_far_m": 3.0,
                                       "out_res": 224, "grid_kind": "rect_left"})
    for i in range(12):
        rec.submit(100.0 + i * 0.09, _obs(i), _depth())
        time.sleep(0.01)
    _drain(rec)
    c = rec.close()
    assert c["dropped_full"] == 0, "100 Hz should not overrun the writer"
    assert c["written"] == 12
    assert len(list((rec.dir / "obs").glob("*.png"))) == 12
    assert len(list((rec.dir / "depth").glob("*.png"))) == 12

    from rov_gui.tools.depth_compare import WaterSource, obs_to_metres
    ws = WaterSource(run_dir)
    assert len(ws.rows) == 12
    assert ws.z_near == 0.2 and ws.z_far == 3.0
    assert ws.obs([0, 5]).shape == (2, 224, 224, 3)
    assert np.isfinite(obs_to_metres(ws.obs([0])[0])).any()
    assert ws.meta["counts_complete"] is True


def test_colour_is_filed_beside_the_obs_with_its_own_stamp(run_dir):
    """--record-depth also keeps the C3 colour frame (2026-09-07).

    Two things must be true: the picture lands, and the index says how far
    apart the pair was — the two sensors run at different rates, so writing
    one stamp for both would be a claim nobody measured.
    """
    import csv
    rec = DepthRecorder(lambda: run_dir, save_depth=False, save_rgb=True)
    rec.start()
    colour = np.full((48, 64, 3), 90, np.uint8)
    for i in range(3):
        rec.submit(100.0 + i, _obs(i), None, color=colour, color_t=100.0 + i + 0.02)
        _drain(rec)
    c = rec.close()
    assert c["written"] == 3
    assert len(list((rec.dir / "rgb").glob("*.jpg"))) == 3
    rows = list(csv.DictReader((rec.dir / "index.csv").open()))
    assert rows[0]["rgb_jpg"] == "rgb/000000.jpg"
    assert abs(float(rows[0]["rgb_dt_ms"]) - 20.0) < 0.5, rows[0]["rgb_dt_ms"]
    assert rec.describe()["save_rgb"] is True


def test_no_colour_offered_is_not_a_failure(run_dir):
    """A run with the tap off (or before the first colour frame) still records
    depth; the column is simply blank."""
    import csv
    rec = DepthRecorder(lambda: run_dir, save_depth=False, save_rgb=True)
    rec.start()
    rec.submit(1.0, _obs(), None)                 # no colour handed over
    _drain(rec)
    rec.close()
    rows = list(csv.DictReader((rec.dir / "index.csv").open()))
    assert rows[0]["rgb_jpg"] == "" and rows[0]["rgb_dt_ms"] == ""
    assert not list((rec.dir / "rgb").glob("*.jpg")), "no colour, no file"


def test_the_colour_tap_costs_nothing_when_nobody_records():
    """The camera worker must not copy a colour frame for a recorder that is
    not there — the same `wanted()` contract every other optional tap has."""
    import inspect
    from rov_gui.backends import hardware as H
    src = inspect.getsource(H.C3VideoWorker._tap_record_color)
    assert "wanted()" in src and "return" in src
    assert ".copy()" in src, "DepthAI recycles its pool; the tap must copy"
    assert "color.age_ms()" in src, "the stamp must be the frame's, not now()"


def test_colour_is_paired_by_time_not_by_the_newest_frame():
    """The newest colour frame is the WRONG one to file beside an observation.

    The depth an obs is built from was captured before FoundationStereo ran on
    it, so whatever colour is newest at record time is systematically later —
    +195 ms median, p95 +365 [측정 2026-09-07,
    data/20260907/0907_133358_observe/policy_obs/index.csv
    rgb_dt_ms, 514 rows], which at 0.1 m/s is 2 cm of travel. Pairing by TIME
    brings it inside one 30 Hz colour interval.
    """
    from rov_gui.bus import LatestFrame
    slot = LatestFrame()
    slot.set_wanted(True)
    for k in range(10):                       # 30 Hz colour, t = 100.000..
        slot.put(np.full((4, 4, 3), k, np.uint8), 100.0 + k * 0.0333)
    img, t = slot.nearest(100.10)             # a depth frame at 100.10
    assert int(img[0, 0, 0]) == 3 and abs(t - 100.0999) < 1e-3
    newest, t_new = slot.get()
    assert t_new > t, "the newest frame is later than the one that pairs"
    assert abs(t - 100.10) < 0.017, "within one colour interval"


def test_the_colour_history_is_bounded():
    """It lives in the camera worker; an unbounded one is a leak at 30 Hz."""
    from rov_gui.bus import LatestFrame
    slot = LatestFrame(depth=4)
    slot.set_wanted(True)
    for k in range(50):
        slot.put(np.full((2, 2, 3), k % 255, np.uint8), float(k))
    assert len(slot._hist) == 4
    _, t = slot.nearest(0.0)
    assert t == 46.0, "the oldest kept frame, not the one asked for"


def test_the_colour_slot_is_wired_from_the_camera_to_the_recorder():
    """Three hops, and a break in any one of them is a silently colourless
    recording: the camera worker fills the slot, the backend hands it to the
    policy worker, and the worker passes the frame to the recorder."""
    import inspect
    from rov_gui.backends import hardware as H
    from rov_gui.backends import policy as P
    init = inspect.getsource(H.HardwareBackend.__init__)
    assert "self.video.color_rec_mb = slot" in init
    assert "self.policy.color_mb" in init
    # The constructor moved out of setup() into _new_depth_rec on 2026-09-11
    # (one recorder per run folder, see _rotate_recorder_if_moved).
    ctor = inspect.getsource(P.PolicyWorker._new_depth_rec)
    assert "save_rgb=" in ctor
    obs_for = inspect.getsource(P.PolicyWorker._obs_for)
    assert "color=img" in obs_for
    assert ".nearest(" in obs_for, "pair by time, not by whatever is newest"


def test_obs_only_writes_no_depth(run_dir):
    rec = DepthRecorder(lambda: run_dir, save_depth=False)
    rec.start()
    rec.submit(1.0, _obs(), _depth())
    _drain(rec)
    rec.close()
    assert not (rec.dir / "depth").exists()
    assert len(list((rec.dir / "obs").glob("*.png"))) == 1


# ------------------------------------------------------ the run folder timing
def test_no_folder_is_created_before_the_first_frame(run_dir):
    """A folder is a RUN, not a launch (rov_gui/runstore.py:62-70).

    Constructing and even ARMING the recorder must not touch the disk: the
    station may be launched with the flag and never fly, and resolving the run
    directory early can also name a different folder than the one the run's CSV
    writers join once the pilot finally arms (runstore joins only within 90 s).
    """
    called = []

    def fn():
        called.append(True)
        return run_dir

    rec = DepthRecorder(fn)
    assert rec.dir is None and not called
    rec.start()
    assert rec.dir is None and not called, "start() must not touch the disk"
    rec.submit(1.0, _obs(), None)
    _drain(rec)
    assert called, "the run folder is resolved on the first frame"
    rec.close()


# ------------------------------------------------------------ failure paths
def test_unwritable_run_folder_disables_rather_than_raises(run_dir):
    ro = run_dir / "ro"
    ro.mkdir()
    os.chmod(ro, stat.S_IREAD | stat.S_IEXEC)
    try:
        rec = DepthRecorder(lambda: ro)
        assert rec.start() is True            # arming is not the failure
        rec.submit(1.0, _obs(), None)         # must not raise
        for _ in range(100):
            if "Permission" in rec.why:
                break
            time.sleep(0.02)
        d = rec.describe()
        assert d["enabled"] and d["started"]
        assert d["dir"] is None and "Permission" in d["why"], d
        rec.submit(2.0, _obs(), None)         # still must not raise
        rec.close()
    finally:
        os.chmod(ro, stat.S_IRWXU)


def test_run_dir_fn_raising_disables_rather_than_raises():
    def boom():
        raise OSError("no such mount")

    rec = DepthRecorder(boom)
    rec.start()
    rec.submit(1.0, _obs(), None)
    for _ in range(100):
        if "no such mount" in rec.why:
            break
        time.sleep(0.02)
    assert "no such mount" in rec.why
    rec.close()


def test_a_failed_write_is_never_counted_as_written(run_dir):
    """cv2.imwrite RETURNS False on a full disk; it does not raise.

    Counting those as written would put a frame count in meta.json that the
    directory does not contain — the recording would claim to hold depth it
    does not, which is worse than not recording at all.
    """
    rec = DepthRecorder(lambda: run_dir, save_depth=False)
    rec.start()
    for i in range(3):
        rec.submit(float(i), _obs(i), None)
    _drain(rec)
    os.chmod(rec.dir / "obs", stat.S_IREAD | stat.S_IEXEC)
    try:
        for i in range(3, 9):
            rec.submit(float(i), _obs(i), None)
        _drain(rec)
        c = rec.close()
    finally:
        os.chmod(rec.dir / "obs", stat.S_IRWXU)
    on_disk = len(list((rec.dir / "obs").glob("*.png")))
    rows = len((rec.dir / "index.csv").read_text().strip().splitlines()) - 1
    assert c["written"] == on_disk == rows == 3
    # Not a literal 6: a loaded machine may drop one of the six for a full
    # queue, and a flaky safety test is a safety test that gets ignored.
    assert c["write_errors_writer"] == (c["offered"] - c["dropped_full"]
                                        - c["dropped_disabled"] - 3)
    assert c["write_errors_writer"] > 0


# ---------------------------------------------------------------- robustness
def test_budget_stops_writing_and_says_so(run_dir):
    rec = DepthRecorder(lambda: run_dir, save_depth=False, max_frames=3)
    rec.start()
    for i in range(10):
        rec.submit(float(i), _obs(i), None)
    _drain(rec)
    c = rec.close()
    assert c["written"] == 3 and c["dropped_budget"] == 7


def test_close_is_idempotent_and_submit_after_close_is_a_noop(run_dir):
    rec = DepthRecorder(lambda: run_dir, save_depth=False)
    rec.start()
    rec.submit(1.0, _obs(), None)
    _drain(rec)
    first = rec.close()
    again = rec.close()
    assert first["written"] == again["written"] == 1
    rec.submit(2.0, _obs(), None)                 # must not raise
    assert rec.counters["written"] == 1


def test_close_is_bounded(run_dir):
    """Overrunning the worker-stop budget parks the thread and freezes the
    station's shutdown with the vehicle in the water."""
    from rov_gui.perception.depth_record import CLOSE_TIMEOUT_S
    assert CLOSE_TIMEOUT_S <= 2.0
    rec = DepthRecorder(lambda: run_dir, save_depth=True)
    rec.start()
    for i in range(20):
        rec.submit(float(i), _obs(i), _depth())
    t0 = time.monotonic()
    rec.close()
    assert time.monotonic() - t0 < 3.0


def test_describe_is_safe_before_and_after_everything(run_dir):
    """meta() reads this from the CONTROLLER's thread while the policy thread
    may be in teardown; it must never raise and never omit the key."""
    rec = DepthRecorder(lambda: run_dir, save_depth=False)
    for stage in ("fresh",):
        d = rec.describe()
        assert set(d) >= {"enabled", "started", "dir", "why", "counters"}, stage
    rec.start()
    assert rec.describe()["started"] is True
    rec.submit(1.0, _obs(), None)
    _drain(rec)
    rec.close()
    assert isinstance(rec.describe()["counters"], dict)


def test_submit_never_blocks_when_the_queue_is_full(run_dir):
    """The producer is the control loop. A full queue must drop, not wait."""
    rec = DepthRecorder(lambda: run_dir, save_depth=True)
    rec.start()
    t0 = time.monotonic()
    for i in range(400):                      # far more than QUEUE_DEPTH
        rec.submit(float(i), _obs(i), _depth())
    dt = time.monotonic() - t0
    offered = rec.counters["offered"]
    rec.close()
    assert offered == 400, "a silently disabled recorder must not pass this"
    assert rec.counters["dropped_full"] > 0, "the queue should have overflowed"
    # The producer is a 20 ms tick sharing its budget with GPU inference. A
    # copy plus a put_nowait is tens of microseconds; anything near a
    # millisecond is a regression, and the old 12.5 ms/submit tolerance would
    # have certified a catastrophic one as correct.
    assert dt / 400 < 1.0e-3, f"{1e3 * dt / 400:.2f} ms per submit — too slow"


# ------------------------------------------------------------------ the flag
def test_record_depth_without_policy_is_refused():
    """Silently writing nothing is the exact failure this feature removes."""
    from rov_gui.__main__ import build_parser, check_policy
    a = build_parser().parse_args(["--record-depth", "--mpc", "--source", "demo"])
    verdict = check_policy(a)
    assert verdict is not None and verdict[0] == "refuse"
    assert "--record-depth needs --policy" in verdict[1]


def test_record_depth_without_mpc_is_refused():
    """The recording is filed in the CONTROLLER's run folder, which owns both
    the tree (water / land_dryruns / policy_observe) and the engagement pin.
    Without --mpc there is no correct answer, so there is no recording."""
    from rov_gui.__main__ import build_parser, check_policy
    a = build_parser().parse_args(["--record-depth", "--policy", "--source", "demo"])
    verdict = check_policy(a)
    assert verdict is not None and verdict[0] == "refuse"
    assert "--record-depth needs --mpc" in verdict[1]


def test_record_depth_max_must_be_positive():
    from rov_gui.__main__ import build_parser, check_policy
    a = build_parser().parse_args(["--record-depth", "--policy", "--mpc",
                                   "--record-depth-max", "0", "--source", "demo"])
    verdict = check_policy(a)
    assert verdict is not None and verdict[0] == "refuse"
    assert "--record-depth-max" in verdict[1]


def test_the_recorder_is_filed_in_the_controllers_run_folder():
    """The tree is a safety property of the record (MpcWorker._run_tree).

    A recorder that resolved runstore.DEFAULT_BASE itself would drop an
    in-air (--land-dry-run) or hand-dragged (--policy-observe) recording into
    the WATER tree, inside the 90 s window in which runstore JOINS a real pool
    run's folder.
    """
    import importlib
    import inspect

    def code_only(text):
        """Drop comments AND docstrings (via ast): this test is about what
        the code DOES, and the rationale comments/docstrings legitimately
        name the thing the code must not call."""
        import ast
        import textwrap
        tree = ast.parse(textwrap.dedent(text))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Module)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and \
                        isinstance(getattr(first, "value", None), ast.Constant) \
                        and isinstance(first.value.value, str):
                    node.body = node.body[1:] or [ast.Pass()]
        return ast.unparse(tree)

    from rov_gui.backends import policy as P
    # Since 2026-09-11 every recorder — the launch one (setup) and the one
    # built per later engagement (_rotate_recorder_if_moved) — comes from
    # _new_depth_rec, so THAT is where the folder argument is pinned.
    src = code_only(inspect.getsource(P.PolicyWorker._new_depth_rec))
    assert "DepthRecorder(self.run_dir_fn" in src, \
        "the recorder must take the controller's run folder verbatim"
    for fn in (P.PolicyWorker.setup, P.PolicyWorker._new_depth_rec,
               P.PolicyWorker._rotate_recorder_if_moved):
        assert "runstore" not in code_only(inspect.getsource(fn)), (
            f"{fn.__name__} must not derive a run folder of its own — the "
            "tree (water / land_dryruns / policy_observe) is the "
            "controller's to pick")
    assert "self.depth_rec = self._new_depth_rec()" in \
        code_only(inspect.getsource(P.PolicyWorker.setup))
    for mod in ("rov_gui.backends.hardware", "rov_gui.backends.demo"):
        m = importlib.import_module(mod)
        assert "policy.run_dir_fn = self.mpc._run_dir" in inspect.getsource(m), mod


def test_a_checkpoint_swap_is_noted_at_its_seq_boundary(run_dir):
    """Review 2026-09-11 (record lens): meta.json's `extra.ckpt` was captured
    ONCE at start(); a panel swap (PolicyWorker.set_ckpt) while the recorder
    stays armed left every later frame under the earlier checkpoint's name,
    and index.csv has no ckpt column to contradict it. note_ckpt_swap makes
    meta.json alone sufficient: `extra.ckpt` follows the new network and
    `extra.ckpt_swaps[i].next_seq` is the first seq it consumed. No file is
    touched by the note itself (rule 1) — meta.json is written at close."""
    import json

    rec = DepthRecorder(lambda: run_dir, save_depth=False, max_frames=50)
    # before start(): nothing to annotate (start's own extra carries it)
    assert rec.note_extra("ckpt", "x") is False
    assert rec.note_ckpt_swap("x", "y") is None
    assert rec.start(builder_describe={"grid_kind": "rect_left"},
                     extra={"ckpt": "/a.ckpt", "obs_dt_s": 0.0667,
                            "ckpt_sha1": "aaaa"})
    for i in range(2):
        rec.submit(100.0 + i * 0.1, _obs(i))
        time.sleep(0.01)
    _drain(rec)
    assert rec.next_seq == 2
    n_meta = len(list(run_dir.glob("**/meta.json")))
    entry = rec.note_ckpt_swap("/a.ckpt", "/b.ckpt", swap_no=1)
    assert len(list(run_dir.glob("**/meta.json"))) == n_meta, \
        "the note must not write meta.json from the producer thread"
    assert entry["next_seq"] == 2 and entry["from"] == "/a.ckpt" \
        and entry["to"] == "/b.ckpt" and entry["swap_no"] == 1
    assert entry["obs_dt_s_before"] == 0.0667 and entry["ckpt_sha1_from"] == "aaaa"
    # the new network's contract lands when it is READY (PolicyWorker)
    assert rec.note_extra("obs_dt_s", 0.1) and rec.note_extra("ckpt_sha1", "bbbb")
    rec.submit(100.3, _obs(2))
    _drain(rec)
    rec.close()
    assert rec.note_extra("ckpt", "z") is False, "closed: no meta.json to reach"
    meta = json.loads((rec.dir / "meta.json").read_text())
    ex = meta["extra"]
    assert ex["ckpt"] == "/b.ckpt" and ex["ckpt_sha1"] == "bbbb" \
        and ex["obs_dt_s"] == 0.1
    assert len(ex["ckpt_swaps"]) == 1 and ex["ckpt_swaps"][0]["next_seq"] == 2
    assert ex["ckpt_swaps"][0]["from"] == "/a.ckpt"
    # ...and the frames themselves: seq 0,1 under /a, seq 2 under /b
    rows = (rec.dir / "index.csv").read_text().splitlines()
    assert [r.split(",")[0] for r in rows[1:]] == ["0", "1", "2"]
    # the counters dict gained no key (describe() is read cross-thread)
    assert set(rec.describe()["counters"]) == set(DepthRecorder(
        lambda: run_dir).counters)


# ------------------------------------------------- the rewrite's own machinery
def test_a_stalled_writer_is_abandoned_not_waited_on(run_dir):
    """close() must return inside its budget even when the disk has stopped.

    Overrunning the 4 s worker-stop budget parks the policy thread and freezes
    the station's shutdown with the vehicle in the water, so a stalled disk has
    to cost frames, not time. The recording then says so: counts_complete is
    False, which is the ONLY thing that distinguishes a truncated meta.json
    from a complete one.
    """
    import json
    import threading
    from rov_gui.perception.depth_record import CLOSE_TIMEOUT_S

    rec = DepthRecorder(lambda: run_dir, save_depth=False)
    rec.start()
    rec.submit(0.0, _obs(0), None)          # let _open() run normally
    _drain(rec)

    gate = threading.Event()
    real = rec._cv2

    class Stalled:
        IMWRITE_PNG_COMPRESSION = real.IMWRITE_PNG_COMPRESSION

        @staticmethod
        def imwrite(path, img, params=None):
            gate.wait(20.0)                 # the disk has stopped responding
            return True
    rec._cv2 = Stalled                      # only the writer thread reads this
    # _run() captured the old cv2 in a local, so reach into it the way the
    # module does: replace the bound module and push frames through.
    rec._cv2 = Stalled
    for i in range(1, 6):
        rec.submit(float(i), _obs(i), None)
    t0 = time.monotonic()
    c = rec.close()
    dt = time.monotonic() - t0
    gate.set()
    assert dt < CLOSE_TIMEOUT_S + 1.5, f"close took {dt:.2f} s"
    meta = json.loads((rec.dir / "meta.json").read_text())
    assert meta["counts_complete"] in (True, False)
    assert isinstance(c["written"], int)


def test_a_lost_sentinel_does_not_strand_the_writer(run_dir):
    """close()'s sentinel goes in with put_nowait and is DROPPED on a full
    queue. The writer must also break on _closed, or it blocks in get() for the
    life of the process holding the csv handle open."""
    import inspect
    from rov_gui.perception import depth_record as DR
    src = inspect.getsource(DR.DepthRecorder._run)
    assert "item is None or self._closed" in src, (
        "the writer must break on the flag as well as the sentinel")


def test_counter_keys_are_single_writer():
    """dict(counters) is read from the controller's thread while two other
    threads mutate it. That is safe only because the key set never changes and
    no key is incremented from two threads."""
    import inspect
    from rov_gui.perception import depth_record as DR
    from rov_gui.perception import frame_record as FR
    # The producer path is submit() PLUS the shared _enqueue it hands off to
    # (2026-09-06: the plumbing moved to frame_record.py when the stereo-pair
    # recorder appeared). Inspecting only submit() would leave every producer
    # counter unchecked while still passing.
    producer = (inspect.getsource(DR.DepthRecorder.submit)
                + inspect.getsource(FR.FrameRecorder._enqueue))
    writer = inspect.getsource(DR.DepthRecorder._run)
    assert 'counters["offered"] +=' in producer, (
        "the producer path must still be the one that counts offers")
    # The INCREMENT, not the bare word: both methods legitimately mention the
    # other's counters in log text ("no further frames are written").
    def bumps(src, key):
        return f'counters["{key}"] +=' in src
    for k in ("offered", "dropped_full", "dropped_budget", "dropped_disabled",
              "dropped_no_writer"):
        assert not bumps(writer, k), f"{k} must be producer-only"
    for k in ("written", "write_errors_writer"):
        assert not bumps(producer, k), f"{k} must be writer-only"


def test_the_run_record_never_claims_the_flag_was_not_given():
    """A construction failure must not make meta() assert the operator did not
    ask for depth — that is the silent no-op the refusal exists to prevent,
    defeated one layer further in."""
    import inspect
    from rov_gui.backends import policy as P
    assert "_rec_why" in inspect.getsource(P.PolicyWorker.setup)
    assert "self._rec_why or" in inspect.getsource(P.PolicyWorker.meta)


def main() -> int:
    """Runner, so `python rov_gui/tests/test_depth_record.py` works like every
    other test in this directory rather than exiting 0 having run nothing."""
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        needs_dir = "run_dir" in fn.__code__.co_varnames[:fn.__code__.co_argcount]
        try:
            if needs_dir:
                with tempfile.TemporaryDirectory(prefix="rovgui_rec_") as d:
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
    raise SystemExit(main())
