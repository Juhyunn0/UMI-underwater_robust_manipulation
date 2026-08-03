#!/usr/bin/env python3
"""
discover_c3.py — find the C3 on the LAN and prove the desktop can talk to it.

Run this first, and run it again whenever the streamer misbehaves. It separates
the three failures that all look the same from the application's point of view:

    the network is broken            -> TCP 11490 does not open
    broadcast discovery is blocked   -> TCP opens but no UDP answer
    something else owns the camera   -> discovered, state X_LINK_BOOTED

Usage
-----
    python c3_camera/discover_c3.py                     # report
    python c3_camera/discover_c3.py --probe             # also connect and dump
                                                        # sensors + calibration
    python c3_camera/discover_c3.py --probe --json      # machine readable

`--probe` briefly connects with no pipeline. It is harmless, but it does claim
the device for a second or two, so do not run it while the streamer is running.

Exit codes (for scripting)
--------------------------
    0  target found and free to connect
    3  target found but already owned by another pipeline
    4  target not found by discovery (TCP may still work — see the report)
    5  target not reachable on the network at all
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import depthai as dai

from c3_camera import config as C
from c3_camera import device as D

EXIT_OK, EXIT_OWNED, EXIT_NOT_FOUND, EXIT_UNREACHABLE = 0, 3, 4, 5


def local_route_to(ip: str) -> tuple[str | None, str]:
    """Which local address the kernel would use to reach `ip`.

    A UDP socket that is 'connected' sends nothing, so this is a pure routing
    lookup — and it catches the common mistake of the camera being reachable
    over the wrong interface (e.g. a VPN) or not routable at all.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((ip, 9))
            return s.getsockname()[0], "ok"
        finally:
            s.close()
    except OSError as e:
        return None, f"no route: {e}"


def report(args) -> tuple[int, dict]:
    out: dict = {"target": {"ip": args.ip, "mxid": args.mxid}}
    print("=" * 74)
    print(f"C3 discovery — depthai {dai.__version__}, python {sys.version.split()[0]}")
    print("=" * 74)

    # ---------------------------------------------------------- 1. routing
    local, note = local_route_to(args.ip)
    out["local_address"] = local
    print(f"\n[1] host route to {args.ip}")
    if local:
        print(f"    local address : {local}")
        if not local.startswith("192.168.2."):
            print(f"    NOTE: {local} is not on the camera's 192.168.2.x subnet; "
                  "traffic may be leaving over the wrong interface")
    else:
        print(f"    {note}")

    # ------------------------------------------------------ 2. reachability
    reachable, tcp_note = D.tcp_reachable(args.ip, timeout_s=args.timeout)
    out["tcp"] = {"reachable": reachable, "note": tcp_note}
    print(f"\n[2] XLink TCP port")
    print(f"    {tcp_note}")
    if not reachable:
        print("    The camera is not answering. Check PoE power, the GigaBlox")
        print("    switch, the fibre link, and `ip -brief addr` on this host.")

    # --------------------------------------------------------- 3. discovery
    print(f"\n[3] UDP discovery (all devices, any state)")
    found = D.discover(timeout_s=args.timeout)
    out["discovered"] = [
        {"name": f.name, "mxid": f.mxid, "state": f.state,
         "protocol": f.protocol, "platform": f.platform, "free": f.is_free}
        for f in found
    ]
    if not found:
        print("    none found.")
        print("    Broadcast may be blocked by a switch, a firewall, or an")
        print("    unrelated VPN interface. Connecting by --ip does not need")
        print("    discovery, so the streamer can still work.")
    for f in found:
        print(f"    {f}")

    print(f"\n[4] DepthAI-connectable devices")
    avail = D.discover_available()
    out["available"] = [{"name": f.name, "mxid": f.mxid, "state": f.state} for f in avail]
    if not avail:
        print("    none — every discovered device is owned or unreachable.")
    for f in avail:
        print(f"    {f.name:<18} mxid={f.mxid:<18} {f.state}")

    # ------------------------------------------------------------ 5. target
    print(f"\n[5] target")
    target = D.find_c3(ip=args.ip, mxid=args.mxid)
    if target is None:
        expect = args.mxid or args.ip
        print(f"    {expect} was not returned by discovery.")
        if args.mxid is None and reachable:
            print("    But TCP is open, so try the streamer anyway:")
            print(f"        python c3_camera/c3_stream.py --ip {args.ip}")
        code = EXIT_NOT_FOUND if reachable else EXIT_UNREACHABLE
        out["verdict"] = "not_discovered"
        return code, out

    print(f"    {target}")
    if args.mxid is None and target.mxid.upper() != C.DEFAULT_MXID:
        print(f"    NOTE: mx_id {target.mxid} differs from the expected C3 "
              f"({C.DEFAULT_MXID}) — is this a different camera?")
    out["verdict"] = "free" if target.is_free else "owned"

    if not target.is_free:
        print("\n    The camera already has a pipeline loaded, so DepthAI cannot")
        print("    claim it. On this rig that is the BlueOS Madrona extension:")
        print("        python -m c3_camera.madrona status")
        print("        python -m c3_camera.madrona stop --yes")
        return EXIT_OWNED, out

    # ------------------------------------------------------------- 6. probe
    if args.probe:
        print(f"\n[6] connecting (no pipeline) to read capabilities")
        print(f"    expect ~{D.TYPICAL_POE_BOOT_S:.0f} s — DepthAI uploads the "
              "firmware and boots the device over PoE")
        try:
            # Reuse the DeviceInfo discovery already resolved, so DepthAI does
            # not have to search for the device again.
            with D.open_device(None, ip=args.ip, mxid=args.mxid,
                               retries=args.retries, info=target.info) as dev:
                info = D.describe_device(dev)
                calib = D.describe_calibration(dev)
                out["device"] = info
                out["calibration"] = calib
                _print_probe(info, calib)
        except Exception as e:  # noqa: BLE001
            print(f"    probe failed: {e}")
            out["probe_error"] = str(e)
            return EXIT_OWNED, out
    else:
        print("\n[6] capabilities — re-run with --probe to connect and read them")

    print("\n" + "=" * 74)
    print("Camera is reachable and free. Next:")
    print(f"    python c3_camera/c3_stream.py --ip {args.ip}")
    print("=" * 74)
    return EXIT_OK, out


def _print_probe(info: dict, calib: dict) -> None:
    print(f"    product    : {info['product']}  (name {info['name']})")
    print(f"    mxid       : {info['mxid']}")
    print(f"    bootloader : {info['bootloader']}")
    t = info.get("temperature_c")
    if isinstance(t, dict):
        print(f"    temperature: {t['average']:.1f} C average")
    print(f"    IR drivers : {info['ir_drivers'] or 'none (no dot projector / flood LED)'}")

    imu_name = info.get("imu")
    if imu_name:
        print(f"    IMU        : {imu_name}  firmware {info.get('imu_firmware', '?')}")
        print("                 Camera-rigid, and its report timestamps share the")
        print("                 same dai.Clock domain as the image frames — so this")
        print("                 is the IMU to use for visual-inertial SLAM.")
    else:
        print(f"    IMU        : none detected"
              + (f"  ({info['imu_error']})" if info.get("imu_error") else ""))

    print("\n    cameras")
    for cam in info["cameras"]:
        af = "autofocus" if cam["autofocus"] else "fixed focus"
        print(f"      {cam['socket']}  {cam['sensor']:<8} native "
              f"{cam['native'][0]}x{cam['native'][1]}  "
              f"{'/'.join(cam['types'])}  {af}")
        for cfg in cam["configs"]:
            print(f"          {cfg['size'][0]:>5}x{cfg['size'][1]:<5} "
                  f"{cfg['type']:<6} up to {cfg['fps'][1]:>6.1f} fps")

    print("\n    calibration")
    if "error" in calib:
        print(f"      unavailable: {calib['error']}")
        return
    if imu_name:
        imu_ext = calib.get("imu_extrinsics") or {}
        if imu_ext.get("available"):
            print("      IMU extrinsics: present on device")
        else:
            print("      IMU extrinsics: NOT on the device — T_imu_cam is unknown.")
            print("        Visual-inertial use needs it calibrated (e.g. Kalibr) or")
            print("        measured mechanically. Camera intrinsics below are fine.")
    print(f"      stereo pair : left={calib.get('stereo_left')} "
          f"right={calib.get('stereo_right')}  "
          f"baseline={calib.get('baseline_cm')} cm")
    for sock, k in (calib.get("intrinsics") or {}).items():
        if isinstance(k, str):
            print(f"      {sock}: {k}")
        else:
            print(f"      {sock}: {k['default_size'][0]}x{k['default_size'][1]} "
                  f"fx={k['fx']:.1f} fy={k['fy']:.1f} "
                  f"cx={k['cx']:.1f} cy={k['cy']:.1f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    C.add_connection_args(p)
    p.add_argument("--probe", action="store_true",
                   help="connect with no pipeline and dump sensors + calibration")
    p.add_argument("--timeout", type=float, default=3.0,
                   help="discovery / TCP timeout in seconds (default: %(default)s)")
    p.add_argument("--retries", type=int, default=2,
                   help="connection attempts when probing (default: %(default)s)")
    p.add_argument("--json", action="store_true", help="also print a JSON report")
    p.add_argument("--json-out", type=Path, default=None, help="write the JSON report here")
    a = p.parse_args(argv)

    code, data = report(a)

    if a.json:
        print("\n--- JSON ---")
        print(json.dumps(data, indent=2, default=str))
    if a.json_out:
        a.json_out.write_text(json.dumps(data, indent=2, default=str))
        print(f"\nwrote {a.json_out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
