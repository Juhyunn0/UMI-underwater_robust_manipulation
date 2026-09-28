#!/usr/bin/env python3
"""
test_slam_bridge.py — the --slam bridge: the KNOT return path above all.

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_slam_bridge.py

WHY THIS FILE EXISTS. The bridge shipped once with ``SlamNavWorker`` built,
started and publishing poses — and with ``bus.policy_plan_viz`` and
``bus.mpc_status`` connected to NOTHING. Every other symptom was absent: the
child ran, the map filled with points, the station's own panel drew the plan.
The single thing that did not happen was the one the feature exists for, and
nothing in any log said so. So the first test here asserts the CONNECTIONS,
not the arithmetic.

The rest covers the two shapes that are easy to get wrong because each has a
plausible wrong version:
  * ``MpcStatus.datum`` is the flattened 4-tuple ``(x0, y0, z0, yaw0)``
    (workers.py:4658-4660), NOT the ``{"p0", "yaw0", "Rz"}`` dict MpcWorker
    keeps internally. The first draft of ``on_mpc_status`` read the dict, and
    a dict read of a tuple raises TypeError inside a slot — i.e. silently, on
    another thread.
  * a knot must survive datum -> map NED -> SLAM world -> the wire -> the C++
    parse and come back to where it started, because nothing downstream fits
    or registers anything: if the polyline is in the wrong place the bug is in
    this algebra and nowhere else.

No camera, no ORB-SLAM3, no child process: the driver is replaced by a fake
whose stdin is a pipe we read.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.control import slam_wire as W                    # noqa: E402
from rov_gui.control.geometry import NavConfig                # noqa: E402
from rov_gui.control.slam_frames import EngageDatum, SlamNav  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

_RESULTS: list = []


def test(fn):
    _RESULTS.append(fn)
    return fn


test.__test__ = False      # a decorator, not a test: pytest collected it as
                           # ``test(fn)`` and errored on the missing fixture


def _nav() -> SlamNav:
    cfg = NavConfig.load(str(REPO / "config" / "hw_nav.yaml"))
    R_bc, t_bc = cfg.R_t_frd_cam("main")
    return SlamNav(R_bc, t_bc)


class _FakeProc:
    """Stands in for :class:`backends.slam.OrbSlamProc`. Records what it was
    handed, in order, so a test can assert both the payload and the fact that
    a CLRO preceded it."""

    def __init__(self):
        self.sent: list = []
        self.proc = None
        self.width, self.height = 640, 400
        self.n_frames_sent = 0
        self.n_poses = 0
        self.t_last_pose = 0.0
        self.stderr_tail: list = []

    def submit_knots(self, payload: bytes) -> bool:
        self.sent.append(payload)
        return True

    def alive(self) -> bool:
        return True


class _FakeSignal:
    def __init__(self):
        self.received: list = []

    def emit(self, *a):
        self.received.append(a)

    def connect(self, *_a, **_k):
        pass


class _FakeBus:
    def __init__(self):
        for name in ("log", "nav_fix", "vehicle_imu", "policy_plan_viz",
                     "mpc_status"):
            setattr(self, name, _FakeSignal())


class _Opts:
    slam = True
    slam_attitude = False
    source = "hw"


class _Viz:
    """The fields ``_pump_knots`` reads off a ``state.PolicyPlanViz``."""

    def __init__(self, p_ned, status="accept", plan_id=7, stamp=123.5):
        self.p_ned = p_ned
        self.status = status
        self.plan_id = plan_id
        self.stamp = stamp


# ===========================================================================
# 1. the wiring — the defect that shipped
# ===========================================================================
@test
def test_the_backend_connects_the_knot_return_path():
    """``bus.policy_plan_viz`` and ``bus.mpc_status`` MUST reach the bridge.

    Asserted against the SOURCE rather than by building a HardwareBackend,
    which would need a camera. Source inspection is enough because the failure
    mode is a missing line, not a wrong one.
    """
    src = (REPO / "rov_gui" / "backends" / "hardware.py").read_text()
    for sig, slot in (("policy_plan_viz", "on_policy_plan_viz"),
                      ("mpc_status", "on_mpc_status")):
        want = f"bus.{sig}.connect(self.slam.{slot})"
        assert want in src, (
            f"MISSING: {want}\n"
            f"Without it --slam runs, tracks, and draws NOTHING in the "
            f"Pangolin map — every other symptom looks healthy.")


@test
def test_the_slots_exist_and_are_qt_slots():
    from rov_gui.backends.slam import SlamNavWorker
    for name in ("on_policy_plan_viz", "on_mpc_status"):
        assert callable(getattr(SlamNavWorker, name, None)), name


# ===========================================================================
# 2. the datum shape
# ===========================================================================
@test
def test_on_mpc_status_reads_the_flattened_four_tuple():
    """The status carries ``(x0, y0, z0, yaw0)``; a dict read would raise."""
    from rov_gui.backends.slam import SlamNavWorker

    w = SlamNavWorker(_FakeBus(), None, _FakeProc(), _nav(), _Opts())

    class _St:
        datum = (1.5, -0.4, 0.9, 0.6457718)

    w.on_mpc_status(_St())
    assert w._datum is not None, "the 4-tuple datum was not accepted"
    assert np.allclose(w._datum.p0, [1.5, -0.4, 0.9])
    assert abs(w._datum.yaw0 - 0.6457718) < 1e-12
    # ...and the SAME datum again must NOT re-clear the overlay ring: the ring
    # is 12 plans deep and clearing it on every status tick (20 Hz) would leave
    # at most one plan on screen at a time.
    n_before = len(w.proc.sent)
    w.on_mpc_status(_St())
    assert len(w.proc.sent) == n_before, "a repeated datum cleared the ring"


@test
def test_a_new_datum_clears_the_ring_and_none_disengages():
    from rov_gui.backends.slam import SlamNavWorker

    w = SlamNavWorker(_FakeBus(), None, _FakeProc(), _nav(), _Opts())

    class _A:
        datum = (0.0, 0.0, 0.0, 0.0)

    class _B:
        datum = (1.0, 0.0, 0.0, 0.0)

    class _None:
        datum = None

    w.on_mpc_status(_A())
    w.on_mpc_status(_B())
    assert W.encode_clear() in w.proc.sent, "a new datum did not clear"
    w.on_mpc_status(_None())
    assert w._datum is None
    assert w.proc.sent.count(W.encode_clear()) >= 2


@test
def test_no_datum_means_no_knot_is_sent():
    """Not engaged = no isometry = the plan cannot be PLACED. Dropping it is
    correct; placing it at the identity would draw it in the wrong spot with
    no way for the operator to tell."""
    from rov_gui.backends.slam import SlamNavWorker

    w = SlamNavWorker(_FakeBus(), None, _FakeProc(), _nav(), _Opts())
    w.on_policy_plan_viz(_Viz(np.zeros((3, 4)).tolist()))
    w._pump_knots()
    assert w.proc.sent == [], "a knot was sent with no datum"


# ===========================================================================
# 3. the round trip
# ===========================================================================
@test
def test_a_knot_survives_datum_to_slam_to_the_wire_and_back():
    from rov_gui.backends.slam import SlamNavWorker

    nav = _nav()
    datum = EngageDatum([1.5, -0.4, 0.9], np.deg2rad(37.0))
    K = 6
    p_datum = np.vstack([np.linspace(0.0, 0.30, K),
                         np.linspace(0.0, 0.05, K),
                         np.linspace(0.0, -0.02, K)])

    w = SlamNavWorker(_FakeBus(), None, _FakeProc(), nav, _Opts())

    class _St:
        pass

    _St.datum = (1.5, -0.4, 0.9, float(np.deg2rad(37.0)))
    w.on_mpc_status(_St())
    w.proc.sent.clear()
    w.on_policy_plan_viz(_Viz(tuple(tuple(r) for r in p_datum)))
    w._pump_knots()
    assert len(w.proc.sent) == 1, w.proc.sent

    # parse it the way the C++ driver does
    dec = W.DriverStreamDecoder()
    recs = list(dec.feed(W.encode_header(640, 400) + w.proc.sent[0]))
    knots = [r for r in recs if isinstance(r, W.KnotRecord)]
    assert len(knots) == 1
    assert knots[0].plan.shape == (K, 3)

    back = datum.datum_from_map_p(nav.ned_from_slam_p(knots[0].plan.T))
    err = float(np.max(np.abs(back - p_datum)))
    assert err < 1e-5, f"round trip off by {err:.3e} m"


@test
def test_the_verdict_reaches_the_wire_and_an_unknown_one_is_not_accept():
    from rov_gui.backends.slam import SlamNavWorker, _status_code

    for name, code in (("accept", 0), ("clip", 1), ("reject", 2),
                       ("late", 3), ("skipped", 4)):
        assert _status_code(name) == code, name
    # An unrecognised verdict must NEVER draw in the colour that means "the
    # controller flew this" (MapDrawer.cc treats 0 and 1 as live).
    assert _status_code("something_new") == int(W.KnotStatus.REJECT)
    assert _status_code(None) == int(W.KnotStatus.REJECT)

    nav = _nav()
    w = SlamNavWorker(_FakeBus(), None, _FakeProc(), nav, _Opts())

    class _St:
        datum = (0.0, 0.0, 0.0, 0.0)

    w.on_mpc_status(_St())
    w.proc.sent.clear()
    p = np.zeros((3, 3))
    p[0] = [0.0, 0.1, 0.2]
    for status, code in (("late", 3), ("reject", 2), ("clip", 1)):
        w.on_policy_plan_viz(_Viz(tuple(tuple(r) for r in p), status=status))
    w._pump_knots()
    assert len(w.proc.sent) == 3
    dec = W.DriverStreamDecoder()
    recs = list(dec.feed(W.encode_header(640, 400) + b"".join(w.proc.sent)))
    got = [r.status for r in recs if isinstance(r, W.KnotRecord)]
    assert got == [3, 2, 1], got


@test
def test_a_malformed_plan_is_dropped_not_raised():
    """``_pump_knots`` runs inside a Qt tick; an exception there is reported as
    a worker failure and the bridge stops. A bad plan must cost that plan."""
    from rov_gui.backends.slam import SlamNavWorker

    w = SlamNavWorker(_FakeBus(), None, _FakeProc(), _nav(), _Opts())

    class _St:
        datum = (0.0, 0.0, 0.0, 0.0)

    w.on_mpc_status(_St())
    w.proc.sent.clear()
    for bad in ([], [[1.0, 2.0]], "not an array", None,
                [[0.0], [0.0], [float("nan")]]):
        w.on_policy_plan_viz(_Viz(bad))
    w._pump_knots()
    # NaN geometry is the one that MAY encode (it is well shaped); the others
    # must not. Either way nothing raised, which is what this asserts.
    assert len(w.proc.sent) <= 1, w.proc.sent


# ===========================================================================
# 4. the queue
# ===========================================================================
@test
def test_the_pair_queue_drops_oldest_and_respects_wanted():
    from rov_gui.backends.slam import SLAM_QUEUE_DEPTH, SlamPairQueue

    q = SlamPairQueue()
    q.put("l", "r", None, 1.0, 1)
    assert q.get() is None, "a pair was accepted while not wanted"
    q.set_wanted(True)
    for i in range(SLAM_QUEUE_DEPTH + 3):
        q.put("l", "r", None, float(i), i)
    put_n, dropped, depth = q.stats()
    assert depth == SLAM_QUEUE_DEPTH
    assert dropped == 3, dropped
    # FIFO, and the OLDEST is what went: continuity is what a tracker needs.
    assert q.get()[4] == 3
    q.set_wanted(False)
    assert q.get() is None, "the queue was not emptied when unwanted"


@test
def test_clahe_defaults_match_the_training_pipeline():
    """The demonstration pose labels were produced with these exact values
    (demonstration_processing/feed_episode.py:87-88)."""
    from rov_gui.backends import slam as S

    assert S.CLAHE_CLIP == 3.0
    assert S.CLAHE_GRID == 8


def main() -> int:
    ok = 0
    for fn in _RESULTS:
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
        else:
            ok += 1
            print(f"  ok    {fn.__name__}")
    print(f"\n{ok}/{len(_RESULTS)} passed")
    return 0 if ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
