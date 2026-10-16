#!/usr/bin/env python3
"""test_dvl_legacy_replay.py — ``dvl.selection: legacy`` is the shipped
station, bit for bit, on a RECORDED pool segment (R5 of the 2026-10-16
selection change; DTF = RL_controller logs/eval/dvl_trip_fix_20261015).

Why a recording: the legacy-pinned unit tests in test_dvl.py never feed a
report that a primary-only rule would treat differently AND that the
shipped ladder then uses. A mutant that ran the R3 quality flags under
legacy (a flagged report stopped being the jump gate's reference) passed
every one of them; replayed through the worker it changed px / py / the
commands on 20 rows of this very session [측정 DTF verify/mutation/
r5_legacy_flags_caught_by_replay.txt]. This test is that replay, in the
repo.

The fixture (fixtures/dvl_legacy_replay_202432_20261016.json.gz) holds:

    inputs    the first 60 s of data/20261015/1015_202421/mpc_202432 (the
              operator's 10-15 trip session: an engaged hold with real
              false locks, the innovation gate, re-entries and a latching
              difference trip at 33 s), every fix / IMU / DVL report the
              live worker took and every CSV tick, times relative to the
              CSV start and rounded to 1 us, values to 1e-6 (1e-7 for the
              rotation): the INPUT is the fixture, not the recording
    expected  the digests of THE SAME replay run on the shipped station
              (git HEAD 171516542, before the selection change, which knew
              no `selection` key), made once by DTF fix1/
              make_legacy_fixture.py

and the replay drives the real MpcWorker (test_control._worker: stub
controller, the test yaml, a stubbed clock) with the session's own DVL
block and its live proof, ``selection: legacy``. Compared: per CSV row the
estimator and DVL columns HEAD wrote (pose, tag age, bridge, vel_src, nu,
the tag difference, the DVL's columns, est), per MpcStatus the shipped DVL
fields, every DVL log line (the legacy setup line's allowed
". selection LEGACY (...)" suffix stripped), the aid's trips and the run
meta's DVL row counts. Not compared: the commands (the stub controller's
business) and the seven columns schema 22 appended.

    QT_QPA_PLATFORM=offscreen python -m pytest \\
        rov_gui/tests/test_dvl_legacy_replay.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = (Path(__file__).parent / "fixtures"
           / "dvl_legacy_replay_202432_20261016.json.gz")
#: The legacy setup line's one allowed difference from the shipped line.
LEGACY_SUFFIX = (". selection LEGACY (the 2026-10-01..15 ladder: re-entry, "
                 "innovation gate, latching trips)")
#: The shipped (schema 21) CSV columns the estimator and the DVL write.
CSV_COLS = ("t", "px", "py", "pz", "yaw_deg", "tag_age_s", "bridge_s",
            "bridge_tier", "vel_src", "nu_u", "nu_v", "nu_w", "fd_u", "fd_v",
            "fd_w", "dvl_u", "dvl_v", "dvl_w", "dvl_ok", "dvl_valid",
            "dvl_fom", "dvl_alt_m", "dvl_age_s", "est")
STATUS_FIELDS = ("vel_src", "dvl_state", "dvl_note", "dvl_use", "dvl_proven",
                 "dvl_res_m_s", "dvl_age_s", "dvl_alt_m", "dvl_fom")
#: The fixture's clock base (any value: both stations see the same inputs).
BASE = 10_000.0


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()
                          ).hexdigest()


def load() -> dict:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        return json.load(f)


def replay(fx: dict, selection: str | None = "legacy") -> dict:
    """The fixture's inputs through a real MpcWorker -> the digests.
    ``selection`` None = no `selection` key (the shipped station knew none);
    this function is also how the expected digests were made."""
    import rov_gui.control.workers as W
    from rov_gui.state import Conn, DvlSample, NavFix, Telemetry, VehicleImu
    from rov_gui.tests import test_control as TC

    cfg = dict(fx["dvl_cfg"])
    cfg.update(enabled=True, use=False)
    if selection is not None:
        cfg["selection"] = selection
    clock = [BASE + float(fx["events"][0][1]) - 1.0]
    real_now = W.now
    W.now = lambda: clock[0]
    out: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, _pilots, logs = TC._worker(tmp)
        try:
            w.cfg.dvl = cfg
            w._setup_dvl()
            d = w.dvl
            d.proven, d.proof = True, dict(fx["proof"])   # the live proof
            st = []
            bus.mpc_status.connect(st.append)
            engaged = False
            n_engage = 0
            for e in fx["events"]:
                kind, t = e[0], BASE + float(e[1])
                clock[0] = t
                if kind == "i":
                    _k, _t, t_att, t_imu, stamp, r, p, y, wp, wq, wr, ax, \
                        ay, az = e
                    t_att, t_imu = BASE + t_att, BASE + t_imu
                    w.on_vehicle_imu(VehicleImu(
                        roll=r, pitch=p, yaw=y, p=wp, q=wq, r=wr, ax=ax,
                        ay=ay, az=az, depth_m=0.8, t_att=t_att, t_imu=t_imu,
                        t_baro=t_att, t_att_v=t_att, att_map_t0=t_att - 60.0,
                        conn=Conn.ONLINE, stamp=BASE + stamp))
                elif kind == "f":
                    tc = BASE + float(e[2])
                    pp, R = tuple(e[3:6]), tuple(e[6:15])
                    w.on_nav_fix(NavFix(
                        t_capture=tc, n_tags=2, tag_ids=(25, 26), p_ned=pp,
                        R_ned_body=R, yaw_ned=math.atan2(R[3], R[0]),
                        reproj_rms_px=0.3, conn=Conn.ONLINE, stamp=t))
                elif kind == "d":
                    _k, _t, tv, trx, v, valid, fom, alt, nb = e
                    w.on_dvl(DvlSample(
                        t_valid=BASE + tv, t_rx=BASE + trx,
                        v_dvl=None if v is None else tuple(v),
                        valid=bool(valid), fom=fom, altitude=alt,
                        n_beams=nb, kind="velocity", hz=10.0, source="a50",
                        conn=Conn.ONLINE, stamp=t))
                else:                                       # a CSV tick
                    eng = bool(e[2])
                    w.on_telemetry(Telemetry(armed=True, mode="MANUAL",
                                             conn=Conn.ONLINE, stamp=t))
                    if eng and not engaged:
                        w.set_dvl_use(True)
                        w.on_enable(True)
                        w.set_engaged(True)
                        engaged = bool(w.engaged)
                        n_engage += 1
                        w._warmup_left = 0
                    elif not eng and engaged:
                        w.set_engaged(False)
                        engaged = False
                    w.tick()
            meta = w._run_meta()
            path = w._csv_path
            if w._csv is not None:
                w._close_csv()
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            row_vals = [[r[c] for c in CSV_COLS] for r in rows]
            stv = [[(None if getattr(s, k, None) is None else
                     repr(getattr(s, k)) if isinstance(getattr(s, k), float)
                     else getattr(s, k)) for k in STATUS_FIELDS] for s in st]
            dlog = [[lvl, m.replace(LEGACY_SUFFIX, "")] for lvl, m in logs
                    if "dvl" in m.lower()]
            md = meta["dvl"]
            out = {
                "rows": len(rows),
                "engagements": n_engage,
                "vel_src": dict(Counter(r["vel_src"] for r in rows)),
                "csv_sha256": _h(row_vals),
                "csv_row_sha8": [_h(v)[:8] for v in row_vals],
                "status_n": len(stv),
                "status_sha256": _h(stv),
                "dvl_log": dlog,
                "trips": [int(d.n_trips), d.last_trip or ""],
                "aid": [bool(d.proven), d.tripped or "", d.against or "",
                        int(d.n_monitor), int(d.n_gross),
                        int(d.n_innovation), int(d.n_valid)],
                "meta_rows": md.get("rows"),
                "meta_trips": md.get("trips"),
            }
        finally:
            W.now = real_now
            w.teardown()
    return out


def test_legacy_replays_a_recorded_trip_segment_like_the_shipped_station():
    fx = load()
    exp = fx["expected"]
    got = replay(fx, selection="legacy")
    # readable first: the summary numbers, then the first differing row
    assert got["rows"] == exp["rows"] and got["engagements"] == 1
    assert got["vel_src"] == exp["vel_src"], (got["vel_src"], exp["vel_src"])
    assert got["trips"] == exp["trips"], (got["trips"], exp["trips"])
    if got["csv_sha256"] != exp["csv_sha256"]:
        i = next(k for k, (a, b) in enumerate(zip(got["csv_row_sha8"],
                                                   exp["csv_row_sha8"]))
                 if a != b)
        raise AssertionError(f"CSV row {i} differs from the shipped station "
                             f"(columns {CSV_COLS})")
    assert got["status_sha256"] == exp["status_sha256"]
    assert got["dvl_log"] == exp["dvl_log"], [
        (a, b) for a, b in zip(got["dvl_log"], exp["dvl_log"]) if a != b][:3]
    assert got["aid"] == exp["aid"], (got["aid"], exp["aid"])
    assert got["meta_rows"] == exp["meta_rows"]
    assert got["meta_trips"] == exp["meta_trips"]
    # the segment is worth replaying: the shipped ladder's machinery ran
    assert exp["trips"][0] >= 1 and exp["vel_src"].get("dvl", 0) > 100
    assert exp["vel_src"].get("tag_fd", 0) > 100


if __name__ == "__main__":
    test_legacy_replays_a_recorded_trip_segment_like_the_shipped_station()
    print("ok")
