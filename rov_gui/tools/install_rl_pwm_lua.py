#!/usr/bin/env python3
"""install_rl_pwm_lua.py — put rl_pwm_override.lua on the vehicle over MAVLink (no BlueOS clicking).

    P=~/miniforge3/envs/rovgui-pose/bin/python
    $P rov_gui/tools/install_rl_pwm_lua.py                    # read-only: what the vehicle has + what the station gate says
    $P rov_gui/tools/install_rl_pwm_lua.py --install --yes    # DISARMED only: scripting on, upload, arm the script, reboot

LOW mode RL_PWM drives the eight thrusters through a vehicle-side Lua script (rov_gui/README.md "RL_PWM 모드";
rov_gui/control/rl_policies/pwm10_s1/README.md). This tool does the vehicle half of that set-up.

What --install does, in this order, and nothing else:
  1. refuses unless the autopilot's HEARTBEAT says DISARMED — checked again right before each reboot;
  2. SCR_ENABLE 1 and SCR_HEAP_SIZE >= 100000 where they differ, then reboots the autopilot (scripting is set up at boot);
  3. uploads the script to ``scripts/rl_pwm_override.lua`` over MAVLink FTP (making ``scripts/`` if it is missing) and
     reads it back byte for byte;
  4. SCR_USER1 1 (the script's own on-switch);
  5. reboots the autopilot once more and waits for the script's "rl_pwm_override.lua loaded" message.
Every parameter it changes is printed with its old value and written to
data/<YYYYMMDD>/<MMDD_HHMMSS>_vehicle_setup/rl_pwm_lua_install.json, so it can be put back. It changes NO other parameter:
the rest of the station's gate (MOT_n_DIRECTION, SERVOn_FUNCTION, RC9..16_OPTION, RC_OPTIONS, RC_OVERRIDE_TIME,
SYSID_MYGCS) is REPORTED, never fixed — those decide which way a thruster spins and who may drive it.

Link: BlueOS's "GCS Client Link" endpoint (udpout to the topside's 14550), i.e. --mav udpin:0.0.0.0:14550 — the port
QGroundControl uses, so close QGC first. rov_gui listens on 14551/14552 and may keep running; it sees each autopilot
reboot as a link drop. Opens no camera.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")          # FTP needs MAVLink 2

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pymavlink import mavutil                     # noqa: E402

POLICY_DIR = REPO / "rov_gui/control/rl_policies/pwm10_s1"
SCRIPT = POLICY_DIR / "rl_pwm_override.lua"
REMOTE = "scripts/rl_pwm_override.lua"
LOADED_TEXT = "rl_pwm_override.lua loaded"
HEAP_MIN = 100000
STATION_SYSID = 255                               # rov_gui --cmd-sysid default


def _say(*a):
    print(*a, flush=True)


class Vehicle:
    def __init__(self, url: str, source_component: int = 191):
        self.m = mavutil.mavlink_connection(url, source_system=STATION_SYSID,
                                            source_component=source_component)
        self.texts: list = []
        self.hb = None

    # ---- receive
    def _recv(self, types, timeout):
        """Next message of ``types`` from the AUTOPILOT (component 1); STATUSTEXT and HEARTBEAT are kept on the way."""
        t_end = time.monotonic() + timeout
        want = set(types)
        while time.monotonic() < t_end:
            msg = self.m.recv_match(blocking=True, timeout=0.2)
            if msg is None:
                continue
            mt = msg.get_type()
            if mt == "STATUSTEXT":
                self.texts.append((time.monotonic(), msg.text))
            if msg.get_srcComponent() != 1:
                continue
            if mt == "HEARTBEAT" and msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                self.hb = msg
            if mt in want:
                return msg
        return None

    def wait_autopilot(self, timeout=10.0) -> bool:
        msg = self._recv(["HEARTBEAT"], timeout)
        while msg is not None and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
            msg = self._recv(["HEARTBEAT"], timeout)
        if msg is None:
            return False
        self.m.target_system, self.m.target_component = msg.get_srcSystem(), 1
        return True

    def armed(self) -> bool:
        """From a heartbeat at most ~1 s old."""
        self.hb = None
        msg = self._recv(["HEARTBEAT"], 3.0)
        while msg is not None and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
            msg = self._recv(["HEARTBEAT"], 3.0)
        if msg is None:
            raise RuntimeError("no autopilot heartbeat — cannot tell whether it is armed")
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    # ---- parameters
    def read_params(self, names, rounds=4) -> dict:
        got: dict = {}
        for _ in range(rounds):
            missing = [n for n in names if n not in got]
            if not missing:
                break
            for i in range(0, len(missing), 8):          # Sub-4.5 queues ~20 requests; stay well under
                batch = missing[i:i + 8]
                for n in batch:
                    self.m.mav.param_request_read_send(self.m.target_system, 1, n.encode(), -1)
                t_end = time.monotonic() + 1.5
                while time.monotonic() < t_end and any(n not in got for n in batch):
                    msg = self._recv(["PARAM_VALUE"], 0.3)
                    if msg is not None:
                        pid = msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()
                        got[pid.rstrip("\x00")] = float(msg.param_value)
        return got

    def set_param(self, name: str, value: float) -> bool:
        for _ in range(3):
            self.m.mav.param_set_send(self.m.target_system, 1, name.encode(), float(value),
                                      mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            t_end = time.monotonic() + 2.0
            while time.monotonic() < t_end:
                msg = self._recv(["PARAM_VALUE"], 0.3)
                if msg is None:
                    continue
                pid = (msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()).rstrip("\x00")
                if pid == name and abs(float(msg.param_value) - float(value)) < 1e-3:
                    return True
        return False

    # ---- reboot
    def reboot_autopilot(self, wait_s=150.0) -> bool:
        if self.armed():
            raise RuntimeError("the autopilot is ARMED — refusing to reboot it")
        self.m.mav.command_long_send(self.m.target_system, 1,
                                     mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 0,
                                     1, 0, 0, 0, 0, 0, 0)
        ack = self._recv(["COMMAND_ACK"], 3.0)
        if ack is not None and ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f"reboot refused by the autopilot (COMMAND_ACK result {ack.result})")
        t0 = time.monotonic()
        # wait for the heartbeat to STOP (the reboot happened), then to come back
        gone = False
        while time.monotonic() - t0 < 20.0:
            if self._recv(["HEARTBEAT"], 2.5) is None:
                gone = True
                break
        _say(f"   autopilot {'went down' if gone else 'did not visibly go down (no heartbeat gap)'}; waiting for it ...")
        while time.monotonic() - t0 < wait_s:
            if self.wait_autopilot(5.0):
                _say(f"   autopilot back after {time.monotonic() - t0:.0f} s")
                time.sleep(3.0)                 # let scripting start and print
                return True
        return False

    # ---- FTP
    def ftp(self):
        from pymavlink import mavftp
        return mavftp.MAVFTP(self.m, target_system=self.m.target_system, target_component=1)


def gate_report(params: dict) -> list:
    """The station gate's parameter checks (rov_gui/control/rl_pwm.py PwmGate.check), as a list of problems."""
    spec = json.loads((POLICY_DIR / "obs_spec.json").read_text())
    md = [int(v) for v in spec["output"]["pwm_model"]["motor_direction"]]
    out = []
    g = lambda n: params.get(n)                                              # noqa: E731
    servo = {n: g(f"SERVO{n}_FUNCTION") for n in range(1, 17)}
    bad = {n: servo[n] for n in range(1, 9) if servo[n] is None or int(servo[n]) != 32 + n}
    if bad:
        out.append(f"SERVO1..8_FUNCTION must be 33..40 (Motor1..8); vehicle: {bad}")
    carrier = {n: int(f) for n, f in servo.items() if f is not None
               and (51 + 8 <= int(f) <= 51 + 15 or 140 + 8 <= int(f) <= 140 + 15)}
    if carrier:
        out.append(f"SERVO{sorted(carrier)}_FUNCTION are RC9..16 passthrough {carrier} (light/camera/gripper?) — move "
                   f"them off RC 9..16")
    mot = [None if g(f"MOT_{n}_DIRECTION") is None else int(round(g(f"MOT_{n}_DIRECTION"))) for n in range(1, 9)]
    if mot != md:
        out.append(f"MOT_1..8_DIRECTION {mot} != trained {md}")
    for n, want in (("SCR_ENABLE", 1), ("SCR_USER1", 1)):
        if g(n) is None or int(round(g(n))) != want:
            out.append(f"{n} = {g(n)} (needs {want}; --install sets it)")
    if g("SCR_HEAP_SIZE") is None or g("SCR_HEAP_SIZE") < HEAP_MIN:
        out.append(f"SCR_HEAP_SIZE = {g('SCR_HEAP_SIZE')} (needs >= {HEAP_MIN}; --install sets it)")
    if g("RC_OPTIONS") is None or int(round(g("RC_OPTIONS"))) & 2:
        out.append(f"RC_OPTIONS = {g('RC_OPTIONS')} (bit 1 must be clear)")
    if g("RC_OVERRIDE_TIME") is None or not g("RC_OVERRIDE_TIME") > 0:
        out.append(f"RC_OVERRIDE_TIME = {g('RC_OVERRIDE_TIME')} (must be > 0)")
    opts = {c: g(f"RC{c}_OPTION") for c in range(9, 17) if g(f"RC{c}_OPTION") not in (0.0,)}
    if opts:
        out.append(f"RC9..16_OPTION must be 0; vehicle: {opts}")
    if g("SYSID_MYGCS") is None or int(round(g("SYSID_MYGCS"))) != STATION_SYSID:
        out.append(f"SYSID_MYGCS = {g('SYSID_MYGCS')} (must be the station's {STATION_SYSID})")
    return out


NAMES = ([f"SERVO{n}_FUNCTION" for n in range(1, 17)] + [f"MOT_{n}_DIRECTION" for n in range(1, 9)]
         + ["SCR_ENABLE", "SCR_USER1", "SCR_HEAP_SIZE", "RC_OPTIONS", "RC_OVERRIDE_TIME", "SYSID_MYGCS",
            "SYSID_THISMAV"] + [f"RC{c}_OPTION" for c in range(9, 17)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mav", default="udpin:0.0.0.0:14550")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--yes", action="store_true", help="required with --install")
    ap.add_argument("--out-root", default=str(REPO / "data"))
    a = ap.parse_args(argv)
    if a.install and not a.yes:
        _say("--install changes vehicle parameters and reboots the autopilot twice; add --yes")
        return 2

    v = Vehicle(a.mav)
    _say(f"listening on {a.mav} for the autopilot ...")
    if not v.wait_autopilot(10.0):
        _say("no autopilot heartbeat in 10 s (BlueOS 'GCS Client Link' -> 14550 enabled? QGC holding the port?)")
        return 1
    armed = v.armed()
    _say(f"autopilot sysid {v.m.target_system}: {'ARMED' if armed else 'disarmed'}")
    before = v.read_params(NAMES)
    unread = [n for n in NAMES if n not in before]
    if unread:
        _say(f"not answered: {unread}")
    problems = gate_report(before)
    _say("station gate, parameter part:" + ("" if problems else " all OK"))
    for p in problems:
        _say(f"   - {p}")
    if not a.install:
        return 0

    if armed:
        _say("REFUSED: the autopilot is ARMED. Disarm it (vehicle safe, out of the way) and run again.")
        return 2
    record = {"tool": "rov_gui/tools/install_rl_pwm_lua.py", "when": time.strftime("%Y-%m-%d %H:%M:%S"),
              "script": str(SCRIPT.relative_to(REPO)), "remote": REMOTE, "before": before, "changed": {}}

    def setp(name, value):
        old = before.get(name)
        _say(f"   {name}: {old} -> {value}")
        if not v.set_param(name, value):
            raise RuntimeError(f"{name} did not take {value}")
        record["changed"][name] = {"old": old, "new": value}

    _say("1/4 scripting on")
    need_reboot = False
    if before.get("SCR_ENABLE") is None or int(round(before["SCR_ENABLE"])) != 1:
        setp("SCR_ENABLE", 1); need_reboot = True
    if before.get("SCR_HEAP_SIZE") is None or before["SCR_HEAP_SIZE"] < HEAP_MIN:
        setp("SCR_HEAP_SIZE", HEAP_MIN); need_reboot = True
    if need_reboot:
        _say("   rebooting the autopilot (scripting starts at boot) ...")
        if not v.reboot_autopilot():
            _say("the autopilot did not come back within 150 s"); return 1
    else:
        _say("   already on")

    _say("2/4 upload over MAVLink FTP")
    ftp = v.ftp()
    r = ftp.cmd_list(["scripts"])
    if r.error_code:
        _say("   scripts/ not listed — creating it")
        ftp.cmd_mkdir(["scripts"])
    r = ftp.cmd_put([str(SCRIPT), REMOTE])
    r = ftp.process_ftp_reply("put", timeout=60)
    if r.error_code:
        r.display_message(); raise RuntimeError("upload failed")
    with tempfile.TemporaryDirectory() as td:
        back = Path(td) / "readback.lua"
        ftp.cmd_get([REMOTE, str(back)])
        r = ftp.process_ftp_reply("get", timeout=60)
        same = back.exists() and back.read_bytes() == SCRIPT.read_bytes()
    _say(f"   read back: {'identical' if same else 'DIFFERENT / missing'} ({SCRIPT.stat().st_size} bytes)")
    if not same:
        raise RuntimeError("the uploaded script does not read back identical")
    record["uploaded_bytes"] = SCRIPT.stat().st_size

    _say("3/4 arm the script")
    if before.get("SCR_USER1") is None or int(round(before["SCR_USER1"])) != 1:
        setp("SCR_USER1", 1)
    else:
        _say("   SCR_USER1 already 1")

    _say("4/4 reboot so the script loads")
    v.texts.clear()
    if not v.reboot_autopilot():
        _say("the autopilot did not come back within 150 s"); return 1
    t_end = time.monotonic() + 20.0
    while time.monotonic() < t_end and not any(LOADED_TEXT in t for _, t in v.texts):
        v._recv(["STATUSTEXT"], 1.0)
    loaded = any(LOADED_TEXT in t for _, t in v.texts)
    lua_msgs = [t for _, t in v.texts if "lua" in t.lower() or "script" in t.lower() or "RLPWM" in t]
    _say(f"   {'LOADED' if loaded else 'no load message seen'}; vehicle messages: {lua_msgs[-6:]}")
    after = v.read_params(NAMES)
    record.update(after=after, loaded_message_seen=loaded, vehicle_messages=lua_msgs,
                  gate_problems_after=gate_report(after))
    _say("station gate, parameter part, now:" + ("" if not record["gate_problems_after"] else ""))
    for p in record["gate_problems_after"] or ["all OK"]:
        _say(f"   - {p}")
    out = Path(a.out_root) / time.strftime("%Y%m%d") / (time.strftime("%m%d_%H%M%S") + "_vehicle_setup")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rl_pwm_lua_install.json").write_text(json.dumps(record, indent=1, default=str))
    _say(f"record: {out / 'rl_pwm_lua_install.json'}")
    return 0 if loaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
