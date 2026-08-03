#!/usr/bin/env python3
"""
device.py — find the C3 on the LAN and open it, without depending on broadcast.

Two independent ways to locate a PoE OAK device:

  1. UDP discovery — DepthAI broadcasts on the local subnets and devices answer.
     Convenient (gives you mx_id and state) but a broadcast can be dropped by a
     switch, a firewall, or a VPN interface.
  2. Direct IP — `dai.DeviceInfo("192.168.2.191")` needs no discovery at all.
     This is the reliable path when you already know the address, and it is what
     the streamer uses by default.

State semantics that matter operationally:

  X_LINK_BOOTLOADER   idle, running its flash bootloader — free to connect
  X_LINK_FLASH_BOOTED idle, booted from flash — free to connect
  X_LINK_BOOTED       a pipeline is already loaded, i.e. someone owns it
                      (on this rig, that someone is the Madrona extension)

A device only accepts one pipeline at a time, so `X_LINK_BOOTED` from another
host is the cause of the classic "Could not connect to device with mx_id" error.
"""

from __future__ import annotations

import contextlib
import os
import socket
import time
from dataclasses import dataclass

import depthai as dai

from . import config as C


# Uploading the firmware to a PoE device and booting it takes this long; it is
# not a hang. Measured ~11.5-12.3 s on this camera, consistently.
TYPICAL_POE_BOOT_S = 12.0


@contextlib.contextmanager
def _env(**overrides: str | None):
    """Temporarily set DepthAI environment variables, then restore them.

    Scoped rather than assigned, so a short discovery timeout cannot leak into
    later connection attempts in the same process. Note that depthai caches
    environment lookups internally after first read, so an override set here may
    simply be ignored — this is hygiene, not a mechanism to rely on. Anything
    that must take effect belongs in the package __init__, before depthai is
    imported (see c3_camera/__init__.py).
    """
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

# XLink's TCP port on a PoE device. Open == the device is powered and on the
# network, whether or not it currently has an owner.
XLINK_TCP_PORT = 11490

FREE_STATES = (
    dai.XLinkDeviceState.X_LINK_BOOTLOADER,
    dai.XLinkDeviceState.X_LINK_FLASH_BOOTED,
    dai.XLinkDeviceState.X_LINK_UNBOOTED,
)


@dataclass
class Found:
    name: str
    mxid: str
    state: str
    protocol: str
    platform: str
    info: dai.DeviceInfo

    @property
    def is_free(self) -> bool:
        return self.info.state in FREE_STATES

    def __str__(self) -> str:
        own = "free" if self.is_free else "OWNED (a pipeline is already loaded)"
        return (f"{self.name:<18} mxid={self.mxid:<18} {self.state:<20} "
                f"{self.protocol:<14} {self.platform:<18} {own}")


# =============================================================================
# Discovery
# =============================================================================
def discover(timeout_s: float | None = None) -> list[Found]:
    """All OAK devices answering UDP discovery, in any state.

    `timeout_s` bounds only this search; it is restored afterwards so it cannot
    shorten the search a later connection performs.
    """
    override = {"DEPTHAI_SEARCH_TIMEOUT": str(int(timeout_s * 1000))} \
        if timeout_s is not None else {}
    with _env(**override):
        # X_LINK_ANY_STATE explicitly: a device that another host already owns
        # reports X_LINK_BOOTED, and we specifically want to see those in order
        # to say "owned" rather than "missing".
        infos = dai.XLinkConnection.getAllConnectedDevices(
            dai.XLinkDeviceState.X_LINK_ANY_STATE)
    return [
        Found(name=i.name, mxid=i.getMxId(), state=i.state.name,
              protocol=i.protocol.name, platform=i.platform.name, info=i)
        for i in infos
    ]


def discover_available() -> list[Found]:
    """Only devices DepthAI considers connectable right now."""
    return [
        Found(name=i.name, mxid=i.getMxId(), state=i.state.name,
              protocol=i.protocol.name, platform=i.platform.name, info=i)
        for i in dai.Device.getAllAvailableDevices()
    ]


def tcp_reachable(ip: str, port: int = XLINK_TCP_PORT, timeout_s: float = 2.0) -> tuple[bool, str]:
    """Is the device's XLink port accepting connections?

    Independent of DepthAI, so it separates "the network is broken" from
    "DepthAI cannot talk to it".
    """
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return True, f"TCP {ip}:{port} open"
    except socket.timeout:
        return False, f"TCP {ip}:{port} timed out after {timeout_s:g}s"
    except OSError as e:
        return False, f"TCP {ip}:{port} failed: {e}"


def find_c3(ip: str | None = None, mxid: str | None = None) -> Found | None:
    """Locate the target camera, preferring the identifier the caller gave."""
    devices = discover()
    if mxid:
        for d in devices:
            if d.mxid.upper() == mxid.upper():
                return d
        return None
    if ip:
        for d in devices:
            if d.name == ip:
                return d
    return devices[0] if len(devices) == 1 else None


# =============================================================================
# Connection
# =============================================================================
class DeviceBusyError(RuntimeError):
    """The camera is reachable but already owned by another pipeline."""


class PipelineConfigError(RuntimeError):
    """The device was reached, but it rejected the pipeline we asked it to run."""


# Substrings that mean "we could not reach or claim the device". Anything else
# coming out of a connection attempt is the device rejecting the pipeline, which
# retrying cannot fix — see _is_connection_error.
_CONNECTION_MARKERS = (
    "failed to connect",
    "could not connect",
    "cannot find any device",
    "no available devices",
    "device already",
    "x_link_device_not_found",
    "x_link_error",
    "x_link_communication_not_open",
    "device not found",
    "timed out",
    "no devices found",
)


def _is_connection_error(msg: str) -> bool:
    """Is this failure about reaching the device, rather than about the pipeline?

    Worth separating: a pipeline that the device rejects (a bad resolution, an
    output size the node will not accept) fails identically on every attempt, so
    retrying just burns three 12-second boots and then blames the network.
    """
    low = msg.lower()
    return any(m in low for m in _CONNECTION_MARKERS)


def device_info_for(ip: str | None = None, mxid: str | None = None) -> dai.DeviceInfo:
    """Build a DeviceInfo without needing discovery to have succeeded.

    `DeviceInfo(str)` decides for itself whether the string is an IP/USB name or
    an mx_id, so both work; an mx_id still has to be resolved by discovery
    internally, whereas an IP is used directly.

    Note we never call `Device.getAnyAvailableDevice()` or `dai.Device(pipeline)`
    without a DeviceInfo. Those search paths fall back to X_LINK_BOOTED devices,
    i.e. they will happily target a camera another process already owns — which
    on this network means stealing the C3 from Madrona by accident. Every
    connection in this package names its device.
    """
    if mxid:
        # XLinkConnection.getDeviceByMxId, not Device.getDeviceByMxId: the latter
        # searches only {UNBOOTED, BOOTLOADER}, so it reports "not found" for a
        # camera that is present but busy — exactly the case we need to describe.
        ok, info = dai.XLinkConnection.getDeviceByMxId(
            mxid, dai.XLinkDeviceState.X_LINK_ANY_STATE)
        if ok:
            return info
        # Fall back to letting DepthAI resolve it, so the error comes from
        # DepthAI with its own message rather than from us.
        return dai.DeviceInfo(mxid)
    if not ip:
        raise ValueError("need either ip or mxid")
    return dai.DeviceInfo(ip)


def open_device(
    pipeline: dai.Pipeline | None,
    ip: str | None = None,
    mxid: str | None = None,
    retries: int = 3,
    retry_delay_s: float = 4.0,
    verbose: bool = True,
    info: dai.DeviceInfo | None = None,
) -> dai.Device:
    """Connect to the camera, with retries for the PoE release window.

    Expect roughly 12 s: DepthAI uploads the firmware over TCP and boots the
    device on every connection. That cost is per run, not per frame.

    After a previous owner drops the link, an RVC2 PoE device takes a few
    seconds to reset and come back in bootloader state, so an immediate
    reconnect legitimately fails. Retrying covers that window.

    Passing pipeline=None opens the device without starting a pipeline, which is
    enough to query capabilities and calibration. Passing a `info` that
    discovery already resolved skips a re-search.
    """
    if info is None:
        info = device_info_for(ip, mxid)
    target = mxid or ip
    last: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            # The maxUsbSpeed overload is the non-deprecated one; the value is
            # irrelevant over PoE but passing it avoids a DeprecationWarning
            # that would otherwise print on every connection.
            if pipeline is None:
                return dai.Device(info, dai.UsbSpeed.SUPER_PLUS)
            return dai.Device(pipeline, info, dai.UsbSpeed.SUPER_PLUS)
        except Exception as e:  # noqa: BLE001 - depthai raises RuntimeError subclasses
            last = e
            msg = str(e)
            if not _is_connection_error(msg):
                # The device answered and refused the pipeline. Report the node's
                # own complaint verbatim — it names the offending node and value.
                raise PipelineConfigError(
                    f"the device rejected this pipeline: {msg}\n"
                    f"  This is a configuration problem, not a connection problem;"
                    f" retrying will not help.\n"
                    f"  Check the resolution/fps/output-size combination against "
                    f"`discover_c3.py --probe`."
                ) from e
            if verbose:
                print(f"  connect attempt {attempt}/{retries} failed: {msg}")
            if attempt < retries:
                # Re-resolve: the device may have changed state meanwhile.
                time.sleep(retry_delay_s)
                try:
                    info = device_info_for(ip, mxid)
                except Exception:
                    pass

    reachable, note = tcp_reachable(ip) if ip else (False, "no ip given")
    hint = _diagnose(str(last), reachable, note, target)
    raise DeviceBusyError(f"could not open {target}: {last}\n{hint}") from last


def _diagnose(err: str, reachable: bool, note: str, target: str | None) -> str:
    lines = [f"  reachability: {note}"]
    low = err.lower()
    if reachable and ("could not connect" in low or "not found" in low or "no available devices" in low):
        lines += [
            "  The camera answers on the network but DepthAI cannot claim it.",
            "  Almost always this means another pipeline already owns it — on this",
            "  rig, the BlueOS Madrona extension.  Free it with:",
            "      python -m c3_camera.madrona stop",
            "  then wait ~5-10 s for the device to reset into bootloader state.",
        ]
    elif not reachable:
        lines += [
            "  The camera is not answering on TCP 11490 at all.  Check the PoE",
            "  link, the GigaBlox switch, and that the desktop still holds",
            "  192.168.2.1 on the fibre NIC (`ip -brief addr`).",
        ]
    elif "x_link_device_not_found" in low or "no device found" in low:
        lines += [
            "  DepthAI's UDP discovery found nothing.  Connect by IP instead of",
            "  mx_id (--ip 192.168.2.191); broadcast can be blocked by a switch,",
            "  a firewall, or an unrelated VPN interface.",
        ]
    lines.append("  Run `python c3_camera/discover_c3.py --probe` for a full report.")
    return "\n".join(lines)


# =============================================================================
# Reporting
# =============================================================================
def describe_device(device: dai.Device) -> dict:
    """Everything worth knowing about a connected device, as plain data."""
    cams = []
    for f in device.getConnectedCameraFeatures():
        cams.append({
            "socket": f.socket.name,
            "sensor": f.sensorName,
            "native": [f.width, f.height],
            "types": [t.name for t in f.supportedTypes],
            "autofocus": bool(f.hasAutofocus),
            "orientation": f.orientation.name,
            "configs": [
                {"size": [c.width, c.height], "type": c.type.name,
                 "fps": [round(c.minFps, 2), round(c.maxFps, 1)]}
                for c in f.configs
            ],
        })

    out = {
        "mxid": device.getMxId(),
        "name": str(device.getDeviceName()),
        "product": str(device.getProductName()),
        "bootloader": str(device.getBootloaderVersion()),
        "cameras": cams,
        "ir_drivers": device.getIrDrivers(),
    }

    # A camera-rigid IMU whose timestamps share the image clock is what makes
    # visual-inertial SLAM possible from this device, so report it explicitly
    # rather than leaving the caller to guess.
    try:
        imu_name = device.getConnectedIMU()
        out["imu"] = imu_name or None
    except Exception as e:  # noqa: BLE001
        out["imu"] = None
        out["imu_error"] = f"{type(e).__name__}: {e}"
    if out.get("imu"):
        try:
            out["imu_firmware"] = str(device.getEmbeddedIMUFirmwareVersion())
        except Exception:  # noqa: BLE001
            out["imu_firmware"] = None
    try:
        t = device.getChipTemperature()
        out["temperature_c"] = {"average": round(t.average, 1), "css": round(t.css, 1),
                                "mss": round(t.mss, 1), "upa": round(t.upa, 1),
                                "dss": round(t.dss, 1)}
    except Exception as e:  # noqa: BLE001
        out["temperature_c"] = f"<unavailable: {e}>"
    return out


def describe_calibration(device: dai.Device) -> dict:
    """Factory stereo calibration, or the reason it is unusable."""
    try:
        calib = device.readCalibration()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}

    out: dict = {}
    try:
        out["stereo_left"] = calib.getStereoLeftCameraId().name
        out["stereo_right"] = calib.getStereoRightCameraId().name
        out["baseline_cm"] = round(calib.getBaselineDistance(), 3)
    except Exception as e:  # noqa: BLE001
        out["stereo"] = f"<unavailable: {e}>"

    per_cam = {}
    for sock in device.getConnectedCameras():
        try:
            K, w, h = calib.getDefaultIntrinsics(sock)
            dist = calib.getDistortionCoefficients(sock)
            per_cam[sock.name] = {
                "default_size": [w, h],
                "fx": round(K[0][0], 2), "fy": round(K[1][1], 2),
                "cx": round(K[0][2], 2), "cy": round(K[1][2], 2),
                "distortion": [round(float(d), 5) for d in dist[:8]],
            }
        except Exception as e:  # noqa: BLE001
            per_cam[sock.name] = f"<unavailable: {type(e).__name__}: {e}>"
    out["intrinsics"] = per_cam

    # IMU-to-camera extrinsics, if the factory wrote any. On this unit it does not
    # exist, and that absence is itself important information for VIO.
    try:
        from .imu import read_imu_extrinsics
        out["imu_extrinsics"] = read_imu_extrinsics(calib, C.SOCKET_COLOR)
    except Exception as e:  # noqa: BLE001
        out["imu_extrinsics"] = {"available": False, "error": str(e)}
    return out
