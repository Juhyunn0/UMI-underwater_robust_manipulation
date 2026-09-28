#!/usr/bin/env python3
"""attitude_axes_probe.py — does THIS firmware read MANUAL_CONTROL s/t?

    python -m rov_gui.tools.attitude_axes_probe            # vehicle DISARMED

The bench gate (design_final D8, gate 3) for the 6-DoF variant
(``engage.attitude_axes``). Sub 4.1.2+ consumes the MAVLink 2 MANUAL_CONTROL
extension axes ``s`` (pitch) and ``t`` (roll) in MANUAL and writes them —
plus the latched trim and 1500 — into RC channels 1 (pitch) and 2 (roll),
which the vehicle reports back as RC_CHANNELS. On 4.1.0 / 4.1.1 the fields
are silently ignored. So, with the vehicle DISARMED (thrusters cannot turn):

    1. preflight (c3_camera/preflight.py, host + vehicle checks) — the
       hardware-preflight-first rule; the camera is not opened;
    2. open the SAME command link the station uses (``--mavlink-out``,
       ``udpin:0.0.0.0:14552``), wait for the autopilot's HEARTBEAT, REFUSE
       if it reports ARMED, ask for AUTOPILOT_VERSION;
    3. record RC_CHANNELS chan1/chan2 BEFORE, then send ``enabled_extensions
       = 0b11`` frames at 10 Hz with s = +A for ``--hold-s``, neutral, then
       t = +A for ``--hold-s``, neutral, recording chan1/chan2 DURING each
       phase and AFTER. Once s/t have been on the wire every neutral frame
       clears them EXPLICITLY (``enabled_extensions = 0b11, s = t = 0``, the
       station sink's rule) and the very last frame is the plain six-argument
       neutral. AUTOPILOT_VERSION / RC_CHANNELS are taken only from the
       (system, component) this probe addresses — BlueOS routes every
       companion component's traffic to the endpoint (2026-09-26 audit);
    4. write ``data/<YYYYMMDD>/<MMDD_HHMMSS>_probe/attitude_axes_probe.json``
       with the per-phase channel medians, the pass/fail verdict, the
       firmware and wire versions, and print the path + sha1 to pin as
       ``engage.attitude_axes.probe``.

What it PROVES: the firmware reads s/t (chan1 moves under s, chan2 under t).
What it does NOT prove: the mixer SIGN of K/M on this vehicle — that is the
armed in-water sign probe (``engage.attitude_axes.sign_probe``), a separate
mission kind. Whether a DISARMED vehicle reflects MANUAL_CONTROL into
RC_CHANNELS at all is [스펙, 미확인]; a FAIL here on a 4.1.2+ firmware is
therefore "run it armed, in water, with the thrusters watched", not proof
of absence — the JSON says so (``verdict_note``).

Every threshold (``--min-delta-us`` 40 us, the expected ``~0.4 * gain * A``
[예측]) is a guess until the first artefact exists. The vehicle is never
armed by this tool; a HEARTBEAT that turns ARMED mid-probe aborts with a
neutral frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.backends.hardware import firmware_version_str  # noqa: E402
from rov_gui.runstore import run_dir  # noqa: E402

#: chan1 = pitch, chan2 = roll on ArduSub's RC map [스펙: ArduSub joystick.cpp
#: rc channel order, unread verbatim — see KNOWN_ISSUES]
CH_PITCH = 1
CH_ROLL = 2
Z_NEUTRAL = 500          # 0..1000 z convention (backends/hardware.py)
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAVLINK_MSG_ID_AUTOPILOT_VERSION = 148
MAV_CMD_REQUEST_MESSAGE = 512


def _median(vals) -> float | None:
    v = [float(x) for x in vals if x is not None]
    return float(np.median(v)) if v else None


class ProbeAbort(RuntimeError):
    """The vehicle armed, the link never spoke, or the dialect is v1."""


def run_probe(master, *, amplitude: int = 300, hold_s: float = 1.0,
              rate_hz: float = 10.0, settle_s: float = 1.0,
              min_delta_us: float = 40.0, target_sys: int = 1,
              target_comp: int = 1, hb_timeout_s: float = 10.0,
              version_wait_s: float = 5.0, clock=time.monotonic,
              sleep=time.sleep, log=print) -> dict:
    """The probe against an OPEN pymavlink connection (or a fake in tests).

    ``master`` needs ``recv_msg()``, ``mav.manual_control_send``,
    ``mav.command_long_send``, ``mavlink20()`` and ``mav.heartbeat_send``.
    Returns the artefact dict; raises :class:`ProbeAbort` when the vehicle is
    armed (before or during) or no autopilot HEARTBEAT arrives."""
    state = {"armed": None, "mode_raw": None, "fw": "", "rc": None, "hb": 0,
             "foreign": 0}

    def from_vehicle(msg) -> bool:
        """The (system, component) pair this probe addresses, and nobody
        else: a companion's AUTOPILOT_VERSION must not become the firmware
        fact, nor its RC_CHANNELS the sample. An absent accessor (a fake
        in tests; every pymavlink message has both) passes."""
        try:
            src_sys = msg.get_srcSystem()
        except AttributeError:
            src_sys = None
        try:
            src_comp = msg.get_srcComponent()
        except AttributeError:
            src_comp = None
        return ((src_sys is None or int(src_sys) == int(target_sys))
                and (src_comp is None or int(src_comp) == int(target_comp)))

    def note(msg) -> None:
        mt = msg.get_type()
        if mt == "HEARTBEAT":
            if msg.get_srcComponent() not in (None, 1):
                return
            state["hb"] += 1
            state["armed"] = bool(int(getattr(msg, "base_mode", 0)) & MAV_MODE_FLAG_SAFETY_ARMED)
            state["mode_raw"] = int(getattr(msg, "custom_mode", 0))
        elif mt in ("AUTOPILOT_VERSION", "RC_CHANNELS"):
            if not from_vehicle(msg):
                state["foreign"] += 1
                return
            if mt == "AUTOPILOT_VERSION":
                state["fw"] = firmware_version_str(getattr(msg, "flight_sw_version", None)) or state["fw"]
            else:
                state["rc"] = tuple(int(getattr(msg, f"chan{i}_raw", 0) or 0) for i in range(1, 9))

    def drain(budget: int = 200) -> None:
        for _ in range(budget):
            try:
                msg = master.recv_msg()
            except Exception:                                    # noqa: BLE001
                return
            if msg is None:
                return
            note(msg)

    def wait_until(pred, timeout_s: float) -> bool:
        t_end = clock() + timeout_s
        while clock() < t_end:
            drain()
            if pred():
                return True
            sleep(0.02)
        return pred()

    # ---- 1. the autopilot must be talking, and DISARMED
    log("waiting for the autopilot HEARTBEAT ...")
    if not wait_until(lambda: state["hb"] > 0, hb_timeout_s):
        raise ProbeAbort("no autopilot HEARTBEAT on the command link — is the "
                         "BlueOS endpoint pushing to this port?")
    if state["armed"]:
        raise ProbeAbort("the vehicle reports ARMED — this probe runs DISARMED "
                         "only (thrusters must not be able to turn)")
    wire_v2 = bool(master.mavlink20())
    if not wire_v2:
        raise ProbeAbort("the command link is MAVLink 1.x — no extension fields "
                         "exist on this dialect (the sink refuses the variant too)")

    # ---- 2. firmware
    try:
        master.mav.command_long_send(target_sys, target_comp, MAV_CMD_REQUEST_MESSAGE,
                                     0, MAVLINK_MSG_ID_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0)
    except Exception as e:                                       # noqa: BLE001
        log(f"AUTOPILOT_VERSION request failed: {e}")
    wait_until(lambda: bool(state["fw"]), version_wait_s)
    log(f"firmware {state['fw'] or '(not answered)'}, wire MAVLink {'2.0' if wire_v2 else '1.0'}")

    n_ext = [0]          # extension frames (s or t nonzero) sent so far

    def send(s: int = 0, t: int = 0) -> None:
        args = (target_sys, 0, 0, Z_NEUTRAL, 0, 0)
        if s or t:
            args = args + (0, 0b11, int(s), int(t))
        elif n_ext[0]:
            # s/t have been on the wire: a neutral clears them EXPLICITLY
            # (the station sink's rule, backends/hardware.py _send), not by
            # hoping the firmware zeroes an absent extension.
            args = args + (0, 0b11, 0, 0)
        master.mav.manual_control_send(*args)
        if s or t:
            n_ext[0] += 1

    def neutral() -> None:
        """Leave the vehicle neutral on EVERY exit: the explicit s = t = 0
        clear (only if an extension frame ever went out), and only then the
        plain six-argument frame — the last thing on the wire is the
        pre-variant neutral."""
        if n_ext[0]:
            send(0, 0)
        master.mav.manual_control_send(target_sys, 0, 0, Z_NEUTRAL, 0, 0)

    def phase(name: str, seconds: float, s: int = 0, t: int = 0) -> dict:
        """Send one frame shape at rate_hz for `seconds`, collecting RC."""
        c1, c2, n_frames = [], [], 0
        t_end = clock() + seconds
        period = 1.0 / max(1.0, rate_hz)
        last_hb = None
        while clock() < t_end:
            drain()
            if state["armed"]:
                # the finally around the phases sends the neutral sequence
                raise ProbeAbort(f"the vehicle ARMED during phase {name!r} — "
                                 f"neutral sent, probe aborted")
            try:
                master.mav.heartbeat_send(6, 8, 0, 0, 0)     # MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID
            except Exception:                                # noqa: BLE001
                pass
            send(s, t)
            n_frames += 1
            if state["rc"] is not None and state["rc"] is not last_hb:
                last_hb = state["rc"]
                c1.append(state["rc"][CH_PITCH - 1])
                c2.append(state["rc"][CH_ROLL - 1])
            sleep(period)
        rec = {"s": int(s), "t": int(t), "frames": n_frames,
               "rc_samples": len(c1),
               "chan1_median_us": _median(c1), "chan2_median_us": _median(c2),
               "chan1_min_us": (min(c1) if c1 else None), "chan1_max_us": (max(c1) if c1 else None),
               "chan2_min_us": (min(c2) if c2 else None), "chan2_max_us": (max(c2) if c2 else None)}
        log(f"  {name:<8} s={s:+5d} t={t:+5d}  chan1 {rec['chan1_median_us']}  "
            f"chan2 {rec['chan2_median_us']}  ({len(c1)} RC samples)")
        return rec

    log("baseline (neutral frames) ...")
    try:
        before = phase("before", settle_s)
        pitch = phase("pitch", hold_s, s=int(amplitude), t=0)
        mid = phase("neutral", settle_s)
        roll = phase("roll", hold_s, s=0, t=int(amplitude))
        after = phase("after", settle_s)
    finally:
        neutral()        # abort or not: explicit clear, then the plain neutral
    if state["foreign"]:
        log(f"{state['foreign']} AUTOPILOT_VERSION/RC_CHANNELS from other "
            f"components ignored (only {target_sys}/{target_comp} counts)")

    def delta(a: dict, b: dict, key: str):
        va, vb = a.get(key), b.get(key)
        return None if va is None or vb is None else float(vb - va)

    d1_pitch = delta(before, pitch, "chan1_median_us")
    d2_pitch = delta(before, pitch, "chan2_median_us")
    d2_roll = delta(mid, roll, "chan2_median_us")
    d1_roll = delta(mid, roll, "chan1_median_us")
    d1_after = delta(before, after, "chan1_median_us")
    d2_after = delta(before, after, "chan2_median_us")
    have_rc = before["rc_samples"] > 0 and pitch["rc_samples"] > 0 and roll["rc_samples"] > 0
    chan1_moved = d1_pitch is not None and abs(d1_pitch) >= min_delta_us
    chan2_moved = d2_roll is not None and abs(d2_roll) >= min_delta_us
    returned = (d1_after is not None and d2_after is not None
                and abs(d1_after) < min_delta_us and abs(d2_after) < min_delta_us)
    passed = bool(have_rc and chan1_moved and chan2_moved and returned)
    verdict = "PASS" if passed else "FAIL"
    if not have_rc:
        note_v = ("no RC_CHANNELS on the command link — nothing to judge; "
                  "check the BlueOS endpoint / message rates")
    elif passed:
        note_v = ("chan1 moved under s and chan2 under t while DISARMED: this "
                  "firmware reads the extension axes. Mixer SIGN still "
                  "unproven (sign_probe).")
    else:
        note_v = ("the channels did not move as expected. Either the firmware "
                  "ignores s/t (< 4.1.2) OR a disarmed vehicle does not "
                  "reflect MANUAL_CONTROL into RC_CHANNELS [스펙 미확인] — "
                  "repeat armed, in water, thrusters watched, before "
                  "concluding.")
    log(f"{verdict}: {note_v}")
    return {
        "tool": "rov_gui.tools.attitude_axes_probe",
        "purpose": "bench gate 3 of design_final D8: does the firmware read MANUAL_CONTROL s/t",
        "armed_during_probe": False,
        "firmware_version": state["fw"],
        "mavlink_wire_version": "2.0" if wire_v2 else "1.0",
        "target": {"sys": int(target_sys), "comp": int(target_comp)},
        "foreign_msgs_ignored": int(state["foreign"]),
        "custom_mode_raw": state["mode_raw"],
        "amplitude": int(amplitude), "hold_s": float(hold_s), "rate_hz": float(rate_hz),
        "min_delta_us": float(min_delta_us),
        "channel_map": {"chan1": "pitch (s)", "chan2": "roll (t)"},
        "phases": {"before": before, "pitch": pitch, "neutral": mid, "roll": roll, "after": after},
        "deltas_us": {"chan1_under_s": d1_pitch, "chan2_under_s": d2_pitch,
                      "chan2_under_t": d2_roll, "chan1_under_t": d1_roll,
                      "chan1_after": d1_after, "chan2_after": d2_after},
        "chan1_moved_under_s": bool(chan1_moved),
        "chan2_moved_under_t": bool(chan2_moved),
        "returned_to_baseline": bool(returned),
        "pass": passed, "verdict": verdict, "verdict_note": note_v,
        "expected_delta_us_note": "~0.4 * gain * amplitude [예측, unmeasured]",
        "sign_proven": False,
    }


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def write_artefact(result: dict, base: str | Path = "data", when: float | None = None) -> Path:
    d = run_dir(base, when=when, kind="probe", join=False)
    out = d / "attitude_axes_probe.json"
    result = dict(result)
    result["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when))
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--mavlink-out", default="udpin:0.0.0.0:14552",
                   help="the station's command link (default: %(default)s)")
    p.add_argument("--cmd-sysid", type=int, default=255, help="our system id (= SYSID_MYGCS)")
    p.add_argument("--target-sysid", type=int, default=1)
    p.add_argument("--amplitude", type=int, default=300, help="s / t value sent, -1000..1000")
    p.add_argument("--hold-s", type=float, default=1.0)
    p.add_argument("--rate-hz", type=float, default=10.0)
    p.add_argument("--settle-s", type=float, default=1.0, help="neutral / baseline phase length")
    p.add_argument("--min-delta-us", type=float, default=40.0,
                   help="RC channel movement that counts as 'read' [예측]")
    p.add_argument("--out", default="data", help="dated run root (rov_gui/runstore.py)")
    p.add_argument("--blueos-host", default="192.168.2.2")
    p.add_argument("--ip", default="192.168.2.2", help="host to route-check in preflight")
    # the preflight flag set every tool shares (c3_camera.preflight.add_preflight_args)
    try:
        from c3_camera import preflight as PF
        PF.add_preflight_args(p)
    except ImportError:
        PF = None
        p.add_argument("--no-preflight", action="store_true")
    a = p.parse_args(argv)
    if not (0 < abs(int(a.amplitude)) <= 1000):
        p.error("--amplitude must be within 1..1000")

    # ---- hardware-preflight-first: host + vehicle checks, no camera
    if PF is not None:
        a.mavlink_transport = "udp"
        a.mavlink = a.mavlink_out
        rc = PF.gate(a, title="attitude axes probe", camera=False, vehicle=True,
                     out_dir=Path(a.out), display=None)
        if rc is not None:
            return rc
    elif not getattr(a, "no_preflight", False):
        print("c3_camera.preflight is not importable; pass --no-preflight to skip", file=sys.stderr)
        return 2

    from pymavlink import mavutil

    master = mavutil.mavlink_connection(a.mavlink_out, source_system=a.cmd_sysid,
                                        source_component=190)
    print(f"command link {a.mavlink_out} open (sysid {a.cmd_sysid}); the vehicle must be DISARMED")
    try:
        result = run_probe(master, amplitude=a.amplitude, hold_s=a.hold_s, rate_hz=a.rate_hz,
                           settle_s=a.settle_s, min_delta_us=a.min_delta_us,
                           target_sys=a.target_sysid)
    except ProbeAbort as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    finally:
        # run_probe already sent the explicit s/t clear + plain neutral on
        # every exit past the HEARTBEAT check; this is the last resort for an
        # exit before it (no extension frame went out — plain is right).
        try:
            master.mav.manual_control_send(a.target_sysid, 0, 0, Z_NEUTRAL, 0, 0)
            master.close()
        except Exception:                                        # noqa: BLE001
            pass
    result["link"] = a.mavlink_out
    out = write_artefact(result, a.out)
    print(f"wrote {out}\n  sha1 {_sha1(out)}\n  pin as engage.attitude_axes.probe: {out}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
