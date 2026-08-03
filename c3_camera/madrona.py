#!/usr/bin/env python3
"""
madrona.py — free the camera from the BlueOS Madrona extension, and give it back.

An OAK device runs one pipeline at a time. While Madrona holds the C3, a direct
DepthAI connection fails with "Could not connect to device with mx_id ...". This
talks to BlueOS's Kraken extension manager to release it.

    python -m c3_camera.madrona status
    python -m c3_camera.madrona stop    --yes     # disable, then wait for release
    python -m c3_camera.madrona start             # hand the camera back
    python -m c3_camera.madrona restart

Endpoints (read off this BlueOS's own OpenAPI spec at
http://192.168.2.2/kraken/v1.0/openapi.json):

    GET  /kraken/v1.0/installed_extensions
    GET  /kraken/v1.0/list_containers
    POST /kraken/v1.0/extension/disable?extension_identifier=marinesitu.madrona
    POST /kraken/v1.0/extension/enable?extension_identifier=marinesitu.madrona
    POST /kraken/v1.0/extension/restart?extension_identifier=marinesitu.madrona

`disable` is persistent: BlueOS records it and will not bring the extension back
on reboot. Nothing else restores it for you, so `stop` prints the `start`
command and refuses to run without `--yes`. If you would rather not touch the
vehicle's configuration, disable it by hand in the BlueOS Extensions page
instead — this script is a convenience, not the only way.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HOST = "192.168.2.2"
MADRONA_ID = "marinesitu.madrona"
TIMEOUT_S = 20.0


# =============================================================================
# Kraken client
# =============================================================================
def _call(host: str, path: str, method: str = "GET",
          params: dict | None = None, timeout: float = TIMEOUT_S):
    url = f"http://{host}/kraken/v1.0{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"{method} {url} failed: {e.reason}\n"
            f"  Is BlueOS reachable?  try: curl -s http://{host}/kraken/v1.0/"
        ) from e
    if not body.strip():
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def installed_extensions(host: str = DEFAULT_HOST) -> list[dict]:
    return _call(host, "/installed_extensions") or []


def containers(host: str = DEFAULT_HOST) -> list[dict]:
    return _call(host, "/list_containers", timeout=40.0) or []


def find_madrona(host: str = DEFAULT_HOST, identifier: str = MADRONA_ID) -> dict | None:
    for e in installed_extensions(host):
        if e.get("identifier") == identifier:
            return e
    return None


def status(host: str = DEFAULT_HOST, identifier: str = MADRONA_ID) -> dict:
    ext = find_madrona(host, identifier)
    running = []
    try:
        for c in containers(host):
            name = (c.get("name") or "").lower()
            image = (c.get("image") or "").lower()
            if "madrona" in name or "madrona" in image:
                running.append({"name": c.get("name"), "status": c.get("status"),
                                "image": c.get("image")})
    except RuntimeError as e:
        running = [{"error": str(e)}]
    return {"extension": ext, "containers": running}


def set_enabled(enabled: bool, host: str = DEFAULT_HOST,
                identifier: str = MADRONA_ID) -> None:
    path = "/extension/enable" if enabled else "/extension/disable"
    _call(host, path, method="POST", params={"extension_identifier": identifier},
          timeout=60.0)


def restart(host: str = DEFAULT_HOST, identifier: str = MADRONA_ID) -> None:
    _call(host, "/extension/restart", method="POST",
          params={"extension_identifier": identifier}, timeout=60.0)


# =============================================================================
# Camera-side confirmation
# =============================================================================
def wait_for_camera(ip: str, want_free: bool, timeout_s: float = 60.0,
                    poll_s: float = 3.0, verbose: bool = True) -> bool:
    """Watch the camera's XLink state until it is free (or owned) as required.

    Stopping the extension is only half the job: the device then resets and
    re-appears in bootloader state a few seconds later, and connecting during
    that window fails. This waits for the state DepthAI needs.
    """
    from . import device as D

    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        found = D.find_c3(ip=ip)
        state = found.state if found else "not discovered"
        if state != last:
            if verbose:
                print(f"  camera state: {state}")
            last = state
        if found is not None and found.is_free == want_free:
            return True
        # Discovery can miss a broadcast; a live XLink port with no discovery
        # answer still tells us the device is at least powered and on the net.
        time.sleep(poll_s)

    _, note = D.tcp_reachable(ip)
    if verbose:
        print(f"  timed out waiting for the camera to become "
              f"{'free' if want_free else 'owned'}; {note}")
    return False


# =============================================================================
# CLI
# =============================================================================
def _print_status(host: str, identifier: str) -> int:
    s = status(host, identifier)
    ext = s["extension"]
    if ext is None:
        print(f"extension {identifier!r} is not installed on {host}")
        print("installed:")
        for e in installed_extensions(host):
            print(f"  {e.get('identifier')}  ({e.get('name')})")
        return 1
    print(f"extension : {ext.get('name')}  [{ext.get('identifier')}]")
    print(f"image     : {ext.get('docker')}:{ext.get('tag')}")
    print(f"enabled   : {ext.get('enabled')}")
    for c in s["containers"]:
        if "error" in c:
            print(f"container : <{c['error']}>")
        else:
            print(f"container : {c['name']}  {c['status']}")
    if not s["containers"]:
        print("container : none running")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m c3_camera.madrona",
        description="Stop or start the BlueOS Madrona extension that owns the C3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("action", choices=("status", "stop", "start", "restart"))
    p.add_argument("--host", default=DEFAULT_HOST, help="BlueOS address (default: %(default)s)")
    p.add_argument("--identifier", default=MADRONA_ID, help="extension id (default: %(default)s)")
    p.add_argument("--camera-ip", default="192.168.2.191", help="camera IP to watch")
    p.add_argument("--yes", action="store_true",
                   help="required for 'stop': confirms changing the vehicle's "
                        "extension configuration")
    p.add_argument("--wait", type=float, default=60.0,
                   help="seconds to wait for the camera to change state (default: %(default)s)")
    p.add_argument("--no-wait", action="store_true", help="do not watch the camera afterwards")
    a = p.parse_args(argv)

    if a.action == "status":
        return _print_status(a.host, a.identifier)

    ext = find_madrona(a.host, a.identifier)
    if ext is None:
        print(f"extension {a.identifier!r} is not installed on {a.host}")
        return 1

    if a.action == "stop":
        if not a.yes:
            print("Refusing to stop without --yes.")
            print()
            print("  This disables the Madrona extension on the vehicle, which is a")
            print("  persistent BlueOS setting: video through BlueOS/Cockpit will stay")
            print(f"  off until you run  python -m c3_camera.madrona start")
            print()
            print("  Re-run with --yes, or disable it by hand in the BlueOS")
            print(f"  Extensions page at http://{a.host}")
            return 2
        print(f"disabling {ext.get('name')} on {a.host} ...")
        set_enabled(False, a.host, a.identifier)
        print("  disabled.  RESTORE LATER WITH: python -m c3_camera.madrona start")
        if not a.no_wait:
            wait_for_camera(a.camera_ip, want_free=True, timeout_s=a.wait)
        return 0

    if a.action == "start":
        print(f"enabling {ext.get('name')} on {a.host} ...")
        set_enabled(True, a.host, a.identifier)
        print("  enabled.  Madrona will reclaim the camera shortly.")
        if not a.no_wait:
            wait_for_camera(a.camera_ip, want_free=False, timeout_s=a.wait)
        return 0

    print(f"restarting {ext.get('name')} on {a.host} ...")
    restart(a.host, a.identifier)
    print("  restart requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
