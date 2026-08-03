#!/usr/bin/env python3
"""
blueos_endpoint.py — give this recorder its own MAVLink feed alongside QGroundControl.

The problem, measured on this rig
---------------------------------
BlueOS routes MAVLink to the surface with an endpoint:

    name="GCS Client Link"  connection_type=udpout  place=192.168.2.1  argument=14550

QGroundControl binds UDP 14550 on the desktop and consumes that stream. A second
process cannot receive those same datagrams:

* Binding 14550 twice fails outright without SO_REUSEPORT, and *with* it Linux
  **load-balances** datagrams between the sockets rather than duplicating them —
  so the recorder would silently steal roughly half of QGC's packets. That is
  worse than not working, because the pilot's link would degrade invisibly.

The fix is to have BlueOS send a second copy to a second port, which is exactly
what its endpoint list is for.

    python -m c3_camera.blueos_endpoint list
    python -m c3_camera.blueos_endpoint add --port 14551      # needs --yes
    python -m c3_camera.blueos_endpoint remove --port 14551

Adding an endpoint rewrites mavlink-router's configuration, so **QGC's link may
blip for a moment**. Do it before you dive, not during. `--persistent` (default)
makes it survive a reboot so you only do this once.

API (read off this BlueOS's own OpenAPI schema at
http://192.168.2.2:8000/v1.0/openapi.json):

    GET    /v1.0/endpoints/     -> list
    POST   /v1.0/endpoints/     -> body is an ARRAY of Endpoint objects
    DELETE /v1.0/endpoints/     -> body is an ARRAY of Endpoint objects
    Endpoint = {name, owner, connection_type, place, argument,
                persistent, protected, enabled, overwrite_settings}
    required: name, owner, connection_type, place
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

DEFAULT_HOST = "192.168.2.2"
BASE = "/v1.0/endpoints/"
OWNER = "c3_camera"
DEFAULT_NAME = "C3 dataset recorder"


def _call(host: str, method: str, body=None, timeout: float = 30.0):
    url = f"http://{host}:8000{BASE}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                headers={"Content-Type": "application/json"}
                                if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{method} {url} failed: {e.reason}") from e
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def list_endpoints(host: str = DEFAULT_HOST) -> list[dict]:
    return _call(host, "GET") or []


def endpoint_body(port: int, place: str = "192.168.2.1",
                  name: str = DEFAULT_NAME, persistent: bool = True) -> dict:
    return {
        "name": name,
        "owner": OWNER,
        # udpout: BlueOS pushes to us, matching how the existing GCS link works,
        # so we need no inbound firewall hole on the vehicle side.
        "connection_type": "udpout",
        "place": place,
        "argument": int(port),
        "persistent": persistent,
        "protected": False,
        "enabled": True,
    }


def add_endpoint(port: int, host: str = DEFAULT_HOST, place: str = "192.168.2.1",
                 name: str = DEFAULT_NAME, persistent: bool = True):
    return _call(host, "POST", [endpoint_body(port, place, name, persistent)])


def remove_endpoint(port: int, host: str = DEFAULT_HOST,
                    place: str = "192.168.2.1") -> object:
    """Remove by matching the live entry, so the body matches what BlueOS holds."""
    for ep in list_endpoints(host):
        if ep.get("argument") == int(port) and ep.get("owner") == OWNER:
            payload = {k: v for k, v in ep.items()
                       if not k.startswith("__")}
            return _call(host, "DELETE", [payload])
    raise RuntimeError(
        f"no endpoint owned by {OWNER!r} on port {port} found; "
        f"run 'list' to see what is configured")


def find_recorder_endpoint(port: int, host: str = DEFAULT_HOST) -> dict | None:
    for ep in list_endpoints(host):
        if ep.get("argument") == int(port) and ep.get("connection_type") == "udpout":
            return ep
    return None


def _print_list(host: str) -> int:
    eps = list_endpoints(host)
    if not eps:
        print(f"no endpoints returned by {host} (is BlueOS reachable?)")
        return 1
    print(f"{'name':<26} {'type':<8} {'place':<16} {'port':>6} {'en':>3} {'persist':>8} owner")
    for e in eps:
        print(f"{str(e.get('name'))[:25]:<26} {str(e.get('connection_type')):<8} "
              f"{str(e.get('place')):<16} {str(e.get('argument')):>6} "
              f"{'y' if e.get('enabled') else 'n':>3} "
              f"{'y' if e.get('persistent') else 'n':>8} {e.get('owner')}")
    print()
    print("QGroundControl consumes the 14550 'GCS Client Link'. For this recorder,")
    print("add a second udpout endpoint on a different port (e.g. 14551).")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m c3_camera.blueos_endpoint",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("list", "add", "remove"))
    p.add_argument("--host", default=DEFAULT_HOST, help="BlueOS address (default: %(default)s)")
    p.add_argument("--port", type=int, default=14551,
                   help="UDP port BlueOS should push to (default: %(default)s)")
    p.add_argument("--place", default="192.168.2.1",
                   help="destination address = this desktop (default: %(default)s)")
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--no-persistent", dest="persistent", action="store_false",
                   default=True, help="do not keep the endpoint across reboots")
    p.add_argument("--yes", action="store_true",
                   help="required for add/remove: confirms changing the vehicle's "
                        "MAVLink routing")
    a = p.parse_args(argv)

    if a.action == "list":
        return _print_list(a.host)

    if not a.yes:
        print(f"Refusing to {a.action} without --yes.")
        print()
        print("  This changes the vehicle's MAVLink routing configuration and may")
        print("  briefly interrupt QGroundControl's link while mavlink-router")
        print("  reloads. Do it before a dive, not during one.")
        print()
        print(f"  Re-run with --yes, or do it by hand in BlueOS:")
        print(f"      http://{a.host}  ->  Vehicle Setup / Autopilot  ->  "
              f"MAVLink Endpoints  ->  add udpout {a.place}:{a.port}")
        return 2

    try:
        if a.action == "add":
            existing = find_recorder_endpoint(a.port, a.host)
            if existing is not None:
                print(f"an endpoint already pushes to {a.place}:{a.port} "
                      f"(name={existing.get('name')!r}); nothing to do")
                return 0
            add_endpoint(a.port, a.host, a.place, a.name, a.persistent)
            print(f"added: udpout -> {a.place}:{a.port}  (persistent="
                  f"{'yes' if a.persistent else 'no'})")
            print(f"  the recorder can now use:  --mavlink udpin:0.0.0.0:{a.port}")
            print(f"  remove it later with:      python -m c3_camera.blueos_endpoint "
                  f"remove --port {a.port} --yes")
        else:
            remove_endpoint(a.port, a.host, a.place)
            print(f"removed the endpoint pushing to {a.place}:{a.port}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
