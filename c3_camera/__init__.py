"""c3_camera — direct DepthAI access to the MarineSitu C3 (OAK-D-W-POE) stereo camera.

Bypasses the BlueOS/Madrona H264+RTSP re-serving path and talks to the camera
over the LAN with DepthAI/XLink instead, which removes the Raspberry Pi from the
video path entirely.

Layout
------
    config.py    StreamConfig + the *probed* hardware capability tables
    device.py    discovery / connection helpers (direct IP, no broadcast needed)
    pipeline.py  builds the on-device pipeline from a StreamConfig
    source.py    C3Source -> RGBDFrame iterator (the ROS integration seam)
    metrics.py   fps / latency / drop accounting
    geometry.py  intrinsics + depth(mm) -> XYZ(m) deprojection
    viz.py       depth colourisation + on-screen HUD
    madrona.py   stop/start the BlueOS Madrona extension that owns the camera

Entry points
------------
    discover_c3.py   find the camera, report capabilities
    c3_stream.py     live RGB + depth view with fps/latency HUD
    c3_bench.py      headless resolution/fps sweep -> CSV

An OAK device accepts exactly one pipeline at a time: Madrona must not hold the
camera while these tools run.  See README.md.

Importing this package does two things before any submodule touches depthai:
it sets PoE-appropriate DepthAI environment defaults, and it refuses to run on
depthai 3.x.  Both are explained below.
"""

import os as _os

# ---------------------------------------------------------------------------
# DepthAI environment defaults, applied BEFORE depthai is imported anywhere.
#
# depthai caches every environment lookup in a process-global map the first time
# it reads one, so setting these after `import depthai` has no effect. Putting
# them in the package __init__ guarantees they land first, because every entry
# point reaches depthai through `c3_camera.*`.
#
# setdefault, not assignment: an explicit value in the caller's environment wins.
# ---------------------------------------------------------------------------
_ENV_DEFAULTS = {
    # This camera is PoE, so skip the USB half of the device search entirely.
    "DEPTHAI_PROTOCOL": "tcpip",
    # Firmware upload + boot over PoE measured ~12 s on this camera, against a
    # 15 s default. That margin is too thin — a slow moment turns into a
    # spurious "could not connect".
    "DEPTHAI_BOOTUP_TIMEOUT": "30000",   # ms (default 15000)
    "DEPTHAI_CONNECT_TIMEOUT": "15000",  # ms (default 5000)
    #
    # XLINK_LEVEL is deliberately NOT set here. XLink defaults to FATAL, which
    # does hide the TCP/socket errors behind a failed PoE connection — but
    # lowering it to warn makes every normal disconnect print a wall of
    # "XLinkCloseStream ... out_link == NULL" teardown errors, which is both
    # alarming and untrue. Opt in only while diagnosing:
    #
    #     DEPTHAI_LEVEL=debug XLINK_LEVEL=debug <script>
    #
    # (Both are needed: DEPTHAI_LEVEL alone leaves the XLink layer silent.)
}
for _k, _v in _ENV_DEFAULTS.items():
    _os.environ.setdefault(_k, _v)


class DepthAIVersionError(ImportError):
    """This package targets the depthai 2.x API and found something else."""


def _require_depthai_v2() -> str:
    """Fail loudly and early on depthai 3.x rather than halfway through a build.

    This matters more than a normal version check because of how v3 fails. It
    keeps `dai.node.ColorCamera` as a shim while removing `dai.node.XLinkOut`,
    `Device.getOutputQueue` and `dai.RawStereoDepthConfig`, so v2 code gets
    several nodes into pipeline construction before breaking — and v3's
    `dai.Pipeline(createImplicitDevice=True)` opens a device merely by being
    constructed, which on this network means it can claim the C3 out from under
    whoever currently has it.

    On this desktop that is a live hazard, not a hypothetical: conda base is
    python 3.12.11 with depthai 3.5.0 installed. Use the venv (see README).
    """
    import depthai as dai

    version = getattr(dai, "__version__", "unknown")
    if not version.startswith("2."):
        raise DepthAIVersionError(
            f"c3_camera targets the depthai 2.x API but found depthai {version} "
            f"at {getattr(dai, '__file__', '?')}.\n"
            f"  depthai 3.x removes XLinkOut / getOutputQueue and opens a device "
            f"just by constructing a Pipeline, so running this code there can "
            f"claim the camera unintentionally.\n"
            f"  Use the dedicated venv instead:\n"
            f"      ~/.venvs/c3-depthai/bin/python <script>\n"
            f"  Create it with:  bash c3_camera/setup_env.sh"
        )
    return version


DEPTHAI_VERSION = _require_depthai_v2()

__all__ = [
    "config",
    "device",
    "pipeline",
    "source",
    "metrics",
    "geometry",
    "viz",
    "madrona",
]

__version__ = "0.1.0"
