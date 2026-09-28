#!/usr/bin/env python3
"""
__main__.py — CLI entry point.

    python -m rov_gui                       synthetic data, opens no hardware
    python -m rov_gui --source hw           C3 + ArduSub (preflight runs first)
    python -m rov_gui --source hw --allow-command      ... and may transmit
    python -m rov_gui --source ros2         rclpy topics
    ./c3 gui                                the wrapper picks the interpreter

Preflight
---------
The hardware path runs this repo's readiness checks *before anything connects*
(``c3_camera.preflight``), which is the standing rule for every tool here that
can open the camera or the vehicle: report what is not ready while it is still
cheap to fix, and never let a half-open device be the diagnostic. The demo and
ros2 paths skip it — they open no hardware, so there is nothing to check.

The flag names below mirror ``c3_camera.preflight.add_preflight_args`` exactly
and are declared locally rather than imported, because importing that module
imports ``c3_camera``, which requires depthai 2.x to be installed — a
requirement the demo path must not inherit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rov_gui.qt import qt_versions          # noqa: E402  (after sys.path fix)


class _Parser(argparse.ArgumentParser):
    """An ArgumentParser that never hands out an unresolved sentinel.

    Two flags default to None so that ``--fstereo`` can change what their
    default IS (argparse cannot express "20 unless another flag is set"). That
    sentinel must not escape: everything downstream reads the namespace with
    ``getattr(opts, name, default)``, which does not fire for a key that
    exists and is None, so a leaked None reaches a format string or an
    arithmetic and fails somewhere far from here. Resolving inside parse_args
    means tests and embedders that build a namespace through this parser get
    the same values ``main()`` does, without having to know the rule.
    """

    def parse_args(self, *a, **kw):
        return resolve_defaults(super().parse_args(*a, **kw))

    def parse_known_args(self, *a, **kw):
        ns, rest = super().parse_known_args(*a, **kw)
        return resolve_defaults(ns), rest


def _size_arg(text: str) -> str:
    """``WxH`` for argparse, refused at launch where a typo is cheap.

    A malformed value used to fall through ``_parse_size`` as None and the run
    flew at --fstereo-scale, with only the startup line's "scale=0.5" to say
    so. Returns the normalised string, which is what the backend parses.
    """
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not WxH (e.g. 224x224)")
    if w < 64 or h < 64:
        raise argparse.ArgumentTypeError(
            f"{text!r}: both sides must be at least 64 px")
    return f"{w}x{h}"


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="python -m rov_gui",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="demo", choices=("demo", "hw", "ros2"),
                   help="where data comes from (default: %(default)s)")

    g = p.add_argument_group("video")
    g.add_argument("--ip", default="192.168.2.191",
                   help="C3 camera IP (default: %(default)s)")
    g.add_argument("--mxid", default=None, help="connect by mx_id instead of IP")
    g.add_argument("--fps", type=float, default=30.0,
                   help="C3 colour fps (default: %(default)s). MJPEG, so its "
                        "cost depends on the scene: 46 kB/frame on a plain one "
                        "and 121 kB/frame on a detailed one, i.e. 11-29 Mbit/s "
                        "at 30 fps. Lowering it buys much less link than it "
                        "looks like it should — see --depth-size.")
    g.add_argument("--isp-scale", default="1/3", metavar="N/D",
                   help="colour output size as a fraction of 1080p (default: "
                        "%(default)s = 640x360; 1/2 = 960x540, 1/4 = 480x270). "
                        "The numerator must be <=16 and the denominator <=63, "
                        "and the encoder wants the WIDTH a multiple of 32 — "
                        "1/2, 1/3 and 1/4 all land on one.")
    g.add_argument("--mjpeg-quality", type=int, default=80,
                   help="colour JPEG quality (default: %(default)s). Measured "
                        "on this camera's own frames at 640x360: q90 = 56.7 "
                        "kB/frame, q85 = 44.8, q80 = 38.0, q75 = 33.1 "
                        "(c3_camera/datasets/dataset_20260805_174544/rgb, "
                        "re-encoded). Quality is the cheapest way to buy link "
                        "back — q90 -> q80 is a third of the bytes.")
    g.add_argument("--depth-fps", type=float, default=None,
                   help="C3 stereo/depth fps, separate from colour because a "
                        "depth frame costs several times a colour one "
                        "(default: 20, or 5 with --fstereo, which has to buy "
                        "link back for the mono pair). Typing a value always "
                        "wins over both defaults.")
    g.add_argument("--depth-size", default="640x360", metavar="WxH",
                   help="depth output size (default: %(default)s). Depth is RAW "
                        "uint16 on the wire, so its cost is exactly w*h*2*fps "
                        "with no scene dependence: 640x360@10 measured 36.9 "
                        "Mbit/s against 36.86 predicted. At 20 fps that is "
                        "73.7 Mbit/s — 82%% of the ~90 Mbit/s C3 link on its "
                        "own — so the colour stream has to live in the "
                        "remaining ~16, which is what --isp-scale and "
                        "--mjpeg-quality are for. Shrink this instead if you "
                        "would rather spend the link on the pilot's view. "
                        "Ignored with --fstereo, which does not stream device "
                        "depth at all.")
    g.add_argument("--no-depth", dest="depth", action="store_false",
                   help="do not compute or stream depth at all — the cheapest "
                        "way to give the colour feed the whole link")
    g.add_argument("--panel2", default="rov",
                   choices=("rov", "stereo", "none"),
                   help="what the middle panel shows: the ROV's own RGB camera "
                        "over BlueOS RTP, or the C3's left mono image "
                        "(default: %(default)s). --fstereo does NOT take this "
                        "panel — learned depth replaces the C3 Depth panel's "
                        "content instead.")
    g.add_argument("--rov-cam-host", default="192.168.2.1",
                   help="address BlueOS pushes the ROV camera RTP to "
                        "(default: %(default)s = this host)")
    g.add_argument("--rov-cam-fps", type=float, default=30.0,
                   help="the rate BlueOS encodes that stream at (default: "
                        "%(default)s; check http://192.168.2.2:6020/streams). "
                        "Used to tell a buffered frame from a live one — see "
                        "RovCamWorker._grab_latest.")
    g.add_argument("--rov-cam-backend", default="auto",
                   choices=("auto", "gstreamer", "ffmpeg"),
                   help="how to receive that stream. gstreamer is what "
                        "QGroundControl uses and is the low-latency path on this "
                        "machine; ffmpeg goes through OpenCV. auto = gstreamer "
                        "when gst-launch-1.0 exists (default: %(default)s)")
    g.add_argument("--rov-cam-size", default="1280x720",
                   help="size GStreamer scales that stream to before handing it "
                        "over (default: %(default)s). 'auto' matches the panel "
                        "instead and relaunches the pipeline when the panel "
                        "changes size, which uses less CPU but gives a softer "
                        "picture in the small slot; a fixed size lets the panel "
                        "downscale, which looks sharper. Capped at the source's "
                        "1920x1080 either way — there is no more detail to have.")
    g.add_argument("--rov-cam-port", type=int, default=5600,
                   help="UDP port of that RTP stream (default: %(default)s). "
                        "QGroundControl also binds 5600; if it is running, add a "
                        "second endpoint udp://192.168.2.1:5601 in BlueOS > "
                        "Video Streams and pass 5601 here.")

    g = p.add_argument_group("vehicle telemetry (receive only)")
    g.add_argument("--mavlink", default="udpin:0.0.0.0:14551",
                   help="MAVLink listen address (default: %(default)s). QGC holds "
                        "14550; add a second BlueOS endpoint for this one — see "
                        "c3_camera/blueos_endpoint.py")
    g.add_argument("--mavlink-transport", default="udp",
                   choices=("udp", "rest", "none"),
                   help="udp = every message; rest = poll MAVLink2Rest, no vehicle "
                        "configuration needed (default: %(default)s)")
    g.add_argument("--mavlink-rest-url", default="http://192.168.2.2:6040")
    g.add_argument("--blueos-host", default="192.168.2.2",
                   help="probed for tether round-trip time (default: %(default)s)")
    g.add_argument("--water-density", type=float, default=997.0,
                   help="kg/m^3 for the pressure->depth DERIVATION; 1025 seawater "
                        "(default: %(default)s)")
    g.add_argument("--no-mavlink-set-rates", dest="mavlink_set_rates",
                   action="store_false",
                   help="do NOT ask the autopilot to raise its message rates. "
                        "Raising them is on by default because ArduSub ships at "
                        "2-3 Hz, which is useless for anything but a dashboard "
                        "light, and the sensor log beside a depth recording is "
                        "only worth keeping at a real rate. It does TRANSMIT "
                        "(MAV_CMD_SET_MESSAGE_INTERVAL) — pass this to keep the "
                        "station strictly passive.")
    g.add_argument("--mavlink-rate", type=int, default=200,
                   help="rate to request for RAW_IMU / SCALED_IMU2 / ATTITUDE "
                        "(default: %(default)s Hz). This is the AUTOPILOT's IMU "
                        "— the C3's BNO086 is a different sensor with its own "
                        "flag, --c3-imu-rate. Measured on this vehicle "
                        "2026-08-06: default 2-3 Hz; request 50 -> 62, 100 -> "
                        "125, 200 -> 208, and 400 or 1000 -> still 208. ~208 Hz "
                        "is the ceiling, and costs about 0.2 Mbit/s.")
    g.add_argument("--c3-imu-rate", type=int, default=200,
                   help="rate to ask the C3's BNO086 for (default: %(default)s "
                        "Hz). Asking for MORE gets less: 200 requested measured "
                        "194 Hz standalone (c3_camera/pipeline.py:67) while 500 "
                        "requested measured 234.9 Hz in this station "
                        "(data/20260809/0809_202407/c3_depth_20260809_202409_"
                        "c3_imu.json) — the extra request costs throughput "
                        "instead of buying it, which is why the default moved "
                        "500 -> 200 on 2026-08-17. Only accel+gyro are "
                        "streamed. (An earlier version of this help said "
                        "ROTATION_VECTOR collapses every stream to ~40 Hz; the "
                        "measured table in pipeline.py:71 says there is no "
                        "penalty at an EQUAL rate — it is the magnetometer, and "
                        "a rotation vector requested SLOWER than accel/gyro, "
                        "that drag the rest down.)")
    g.add_argument("--no-c3-imu", dest="c3_imu", action="store_false",
                   help="do not stream the C3's IMU at all")
    g.add_argument("--imu-dr", choices=("off", "shadow", "control"),
                   default=None,
                   help="IMU dead reckoning: integrate the C3's BNO086 on its "
                        "own from the end of the mission settle and draw the "
                        "result beside the AprilTag pose, which plays ground "
                        "truth. 'shadow' leaves the controller on the tag "
                        "state; 'control' makes the controller fly on the "
                        "ESTIMATE and needs --imu-dr-control as well. Overrides "
                        "hw_mpc.yaml imu_dr.enabled/mode.")
    g.add_argument("--imu-dr-attitude", choices=("gyro", "ahrs", "vehicle"),
                   default=None,
                   help="how the dead reckoner gets roll/pitch/yaw. 'ahrs' "
                        "(the config default) integrates the gyro and levels "
                        "roll/pitch against gravity, yaw gyro-only; 'gyro' is "
                        "pure integration; 'vehicle' borrows roll/pitch from "
                        "the autopilot to isolate position-only error.")
    g.add_argument("--imu-dr-control", action="store_true",
                   help="arm the closed loop when hw_mpc.yaml already says "
                        "imu_dr.mode: control. NOT needed alongside "
                        "`--imu-dr control`, which is explicit on its own — "
                        "this exists so a config file left in that state "
                        "cannot arm a closed loop on dead reckoning with "
                        "nobody asking for it. Either way, know what it "
                        "means: the controller drives the vehicle toward "
                        "where the IMU thinks it is, and this station has no "
                        "geofence, so the pilot's E-STOP is the stop.")
    g.add_argument("--battery-capacity-mah", type=float, default=0.0,
                   help="pack capacity in mAh, e.g. 18000 for the BlueROV2 "
                        "stock Li-ion (default: %(default)s = unknown). Given "
                        "this, the panel shows mAh REMAINING and derives a "
                        "percentage by coulomb counting, which beats guessing "
                        "from voltage. Both are still DERIVED and labelled "
                        "'est' — the real fix is to set BATT_CAPACITY and a "
                        "current monitor on the vehicle, after which ArduSub "
                        "reports battery_remaining itself and this panel "
                        "prefers that.")
    g.add_argument("--battery-cells", type=int, default=4,
                   help="series cell count for the voltage-based estimate "
                        "(default: %(default)s = BlueROV2 stock 4S). Set this "
                        "rather than relying on inference: one voltage cannot "
                        "distinguish an EMPTY 4S pack (12.0 V) from a FULL 3S "
                        "one, and guessing wrong in that direction reports a "
                        "flat battery as nearly full.")
    g.add_argument("--thrusters", type=int, default=8,
                   help="motor count to display (default: %(default)s = Heavy)")

    g = p.add_argument_group(
        "command output — OFF unless --allow-command AND the UI switch")
    g.add_argument("--allow-command", action="store_true",
                   help="build the MAVLink command sink. Even then nothing is "
                        "transmitted until COMMAND ENABLE is switched on in the "
                        "UI. Make sure no other command source (QGroundControl, "
                        "a joystick) is commanding: ArduSub obeys whichever "
                        "MANUAL_CONTROL arrived last.")
    g.add_argument("--mavlink-out", default="udpin:0.0.0.0:14552",
                   help="the command socket (default: %(default)s). udpin binds "
                        "a port BlueOS pushes to and replies to the sender — the "
                        "way QGC commands. It needs the endpoint to exist: "
                        "./c3 blueos_endpoint add --port 14552 --yes . "
                        "udpout:HOST:PORT is also accepted, but note that UDP "
                        "14550 on this vehicle is CLOSED (probed 2026-08-06), so "
                        "sending there silently does nothing.")
    g.add_argument("--target-sysid", type=int, default=1,
                   help="the VEHICLE's system id, i.e. who ARM/DISARM and "
                        "MANUAL_CONTROL are addressed to (default: %(default)s). "
                        "A udpout link never receives, so this cannot be learned "
                        "from a heartbeat and must be stated.")
    g.add_argument("--cmd-sysid", type=int, default=255,
                   help="system id this station sends commands AS. It must equal "
                        "the vehicle's SYSID_MYGCS parameter (255 by default, "
                        "which is why that is the default here) — ArduPilot drops "
                        "MANUAL_CONTROL from any other system id silently, while "
                        "still accepting ARM from it. The GUI reads SYSID_MYGCS "
                        "and turns the CMD pill red on a mismatch.")
    g.add_argument("--deadman-ms", type=int, default=500,
                   help="input older than this sends NEUTRAL (default: %(default)s)")
    g.add_argument("--z-convention", default="0..1000",
                   choices=("0..1000", "-1000..1000"),
                   help="MANUAL_CONTROL z axis convention for your ArduSub "
                        "version — getting it wrong makes the vehicle dive when "
                        "you meant neutral (default: %(default)s)")
    # Defaults are THIS vehicle's assignment, read off QGC > Vehicle Setup >
    # Joystick > Button Assignment and confirmed against the autopilot's own
    # BTNn_FUNCTION parameters on 2026-08-06:
    #
    #     BTN0_FUNCTION  = 77   servo_1_max_momentary   gripper open   (HELD)
    #     BTN15_FUNCTION = 76   servo_1_min_momentary   gripper close  (HELD)
    #     BTN14_FUNCTION = 32   lights1_brighter        one PRESS = one notch
    #     BTN13_FUNCTION = 33   lights1_dimmer
    #
    # They are defaults rather than required flags because this is the rig the
    # station is for. They are still CHECKED at startup: the sink reads each
    # BTNn_FUNCTION back and, if a button does not do what is assumed here, it
    # says so and stops using that bit rather than pressing something unknown on
    # a vehicle it was not configured for.
    for name, default, help_text in (
            # These describe the VEHICLE's BTNn_FUNCTION assignment. They are not
            # a preference: setting them to a button the vehicle has assigned to
            # something else makes the UI's own gripper controls press that other
            # thing instead. To drive the gripper from a different PAD button,
            # use --js-remap; it rewrites what the pad sends without lying about
            # where the vehicle's function actually lives.
            ("gripper-open", 0, "gripper open — expects servo_1_max_momentary "
                                "(77), i.e. a HELD button"),
            ("gripper-close", 15, "gripper close — expects servo_1_min_momentary "
                                  "(76)"),
            ("lights-up", 14, "one notch brighter — expects lights1_brighter "
                              "(32), i.e. one action per PRESS"),
            ("lights-down", 13, "one notch dimmer — expects lights1_dimmer (33)"),
            ("tilt-up", 10, "camera mount up — expects mount_tilt_up (22), a "
                            "HELD button"),
            ("tilt-down", 9, "camera mount down — expects mount_tilt_down (23)"),
            ("tilt-center", 7, "level the camera mount — expects mount_center "
                               "(21), one action per PRESS")):
        g.add_argument(f"--btn-{name}", type=int, default=default,
                       help=f"button number mapped to {help_text}. "
                            f"Default %(default)s is this vehicle's; pass -1 to "
                            f"leave that function alone entirely.")
    g.add_argument("--tilt-servo", type=int, default=-1,
                   help="servo output the camera mount is wired to, for a tilt "
                        "readout when the vehicle does not send MOUNT_STATUS "
                        "(default: %(default)s = off). MOUNT_STATUS is preferred "
                        "and used automatically when present: it is degrees the "
                        "vehicle reports, whereas a PWM has to be mapped through "
                        "the servo range and is only ever [유도].")
    g.add_argument("--tilt-min-deg", type=float, default=-45.0,
                   help="mount angle at minimum PWM, for --tilt-servo only "
                        "(default: %(default)s)")
    g.add_argument("--tilt-max-deg", type=float, default=45.0,
                   help="mount angle at maximum PWM, for --tilt-servo only "
                        "(default: %(default)s)")
    g.add_argument("--tilt-rate-deg-s", type=float, default=None,
                   help="mount slew while UP/DOWN is held, deg/s, for the "
                        "PAYLOAD panel's dead-reckoned tilt readout when the "
                        "vehicle reports no angle (default: hw_nav.yaml "
                        "second_cam.tilt_rate_deg_s, [예측] 30). Time a sweep "
                        "against the tag-measured tilt and set it.")
    g.add_argument("--arm-mode", default="MANUAL",
                   choices=("MANUAL", "STABILIZE", "ALT_HOLD", "ACRO", "keep"),
                   help="flight mode the ARM button asks for BEFORE arming "
                        "(default: %(default)s). ArduSub keeps whatever mode it "
                        "was left in, and in a stabilising mode an armed "
                        "vehicle runs its thrusters with no stick input at all "
                        "— which is how pressing ARM came to start the motors. "
                        "MANUAL is the only mode in which an armed vehicle sits "
                        "still. Pass 'keep' to arm into whatever mode the "
                        "vehicle is already in. This does NOT lock the mode: "
                        "the MANUAL/STAB/DEPTH buttons change it whenever you "
                        "ask, before or after arming.")
    g.add_argument("--lights-servo", type=int, default=13,
                   help="servo output the lights are wired to, so the panel can "
                        "show the level the vehicle REPORTS instead of our press "
                        "count (default: %(default)s — measured on this vehicle, "
                        "1100 us with the lights off). Pass -1 for no feedback.")
    g.add_argument("--lights-steps", type=int, default=8,
                   help="how many presses cover the light's full range; must "
                        "match the vehicle's JS_LIGHTS_STEPS (default: "
                        "%(default)s). Only a starting guess: the GUI reads the "
                        "real value off the vehicle and retunes the slider.")

    g = p.add_argument_group(
        "joystick — read straight from /dev/input/js*, no packages needed")
    g.add_argument("--joystick", default="auto",
                   help="device path, 'auto' (first one found, re-scanned after "
                        "an unplug), or 'none' (default: %(default)s)")
    g.add_argument("--js-deadzone", type=float, default=0.08,
                   help="centre deadzone, rescaled so full deflection still "
                        "reaches 1.0 (default: %(default)s)")
    g.add_argument("--js-scale", type=float, default=1.0,
                   help="multiplier on every stick axis, applied before the "
                        "panel's OUTPUT slider (default: %(default)s)")
    # These DEFAULT TO THE DEVICE, not to a number, and that is deliberate: the
    # same Xbox pad enumerates its axes one way over Bluetooth (2,3 = right
    # stick) and another over USB (3,4 = right stick, 2 = left trigger), so a
    # number baked in here is correct for one cable and silently wrong for the
    # other. Pinned numbers on 2026-08-17 sent the yaw stick to HEAVE and left
    # yaw on a trigger. joystick.py asks the driver instead (JSIOCGAXMAP), and
    # the startup log prints what it found. Pass a number to override.
    for axis in ("surge", "sway", "yaw", "heave"):
        g.add_argument(f"--js-axis-{axis}", type=int, default=None,
                       help=f"pin the axis number driving {axis}; negative "
                            f"inverts. Default is auto-detected from the pad. "
                            f"0 cannot be inverted — use --js-invert-{axis}.")
    for axis in ("surge", "sway", "yaw", "heave"):
        g.add_argument(f"--js-invert-{axis}", action="store_true",
                       help=f"invert {axis}, for when the axis number is 0")
    g.add_argument("--no-js-translate", dest="js_translate", action="store_false",
                   help="send the KERNEL's button numbers instead of "
                        "translating them to the vehicle's (SDL/QGC) numbering. "
                        "Translation is on by default and is not cosmetic: the "
                        "kernel and SDL order buttons differently, so on this "
                        "pad the untranslated left bumper (kernel 6) reaches "
                        "the vehicle as BTN6 = arm. Only use this if you "
                        "configured BTNn_FUNCTION against kernel numbering "
                        "yourself.")
    g.add_argument("--no-js-passthrough", dest="js_passthrough",
                   action="store_false",
                   help="stop forwarding the gamepad's own button bitmask to "
                        "the vehicle. Passthrough is ON by default and is what "
                        "makes the pad behave identically here and in QGC: the "
                        "vehicle's BTNn_FUNCTION parameters decide what each "
                        "button does, so buttons this station has no widget for "
                        "(mode changes, gain, trim, shift, input_hold_set) work "
                        "too. Turning it off leaves only the four functions the "
                        "--js-btn-* flags name, interpreted topside.")
    # These now default to the SAME numbers as --btn-*, i.e. the vehicle's own
    # assignment, because they no longer describe a second mapping — they only
    # say which on-screen chip should light when that button goes down. Having
    # them default to different numbers is exactly what made a pad button do one
    # thing in QGC and another here.
    g.add_argument("--js-remap", default="11:0,12:15", metavar="FROM:TO,...",
                   help="rewrite pad buttons before they are sent, in the "
                        "VEHICLE's numbering (default: %(default)s). This "
                        "exists because a pad cannot always reach every "
                        "function: gripper close lives on button 15, which SDL "
                        "calls MISC1 (\"Share\"), and this pad reports buttons "
                        "0-14 only — so that function was physically "
                        "unpressable. The default puts the gripper on the "
                        "D-pad (11 up -> 0 open, 12 down -> 15 close). The "
                        "rewritten button is sent INSTEAD of the original, so "
                        "one press is still one function. Pass '' to disable.")
    for name, default in (("gripper-close", 15), ("gripper-open", 0),
                          ("lights-down", 13), ("lights-up", 14),
                          ("tilt-down", 9), ("tilt-center", 7), ("tilt-up", 10)):
        g.add_argument(f"--js-btn-{name}", type=int, default=default,
                       help=f"joystick button that lights the {name} indicator "
                            f"(default: %(default)s, the vehicle's own "
                            f"assignment); -1 to leave it unmapped")

    g = p.add_argument_group(
        "learned stereo depth (FoundationStereo) — OFF unless --fstereo")
    g.add_argument("--fstereo", action="store_true",
                   help="make FoundationStereo THE depth source for this run. "
                        "The C3's own stereo depth is not streamed and not "
                        "used: the depth panel, the cursor probe, the "
                        "depth-vs-MAP check, --pose/FoundationPose and the "
                        "recordings all take the learned map. One instrument "
                        "per run, so a run is never a mixture. OFF by default "
                        "— it loads 3 GiB onto the GPU (~6 s) in the process "
                        "that commands the vehicle, and it re-plans the link "
                        "(--depth-fps becomes 15 for the mono pair). Without "
                        "the flag nothing is imported and no frame is copied. "
                        "Needs the rovgui-pose env; './c3 env' shows which "
                        "interpreter the GUI resolved.")
    g.add_argument("--fstereo-repo", default=None, metavar="DIR",
                   help="the FoundationStereo checkout (default: "
                        "$FOUNDATION_STEREO_REPO, else "
                        "~/Desktop/FoundationStereo/FoundationStereo)")
    g.add_argument("--fstereo-ckpt", default=None, metavar="PTH",
                   help="checkpoint (default: <repo>/pretrained_models/"
                        "23-51-11/model_best_bp2.pth, the ViT-large model). "
                        "cfg.yaml must sit beside it.")
    g.add_argument("--fstereo-iters", type=int, default=None,
                   help="GRU refinement iterations (default: 8, or 16 with "
                        "--policy — the training depth store's setting "
                        "[측정: ~/Desktop/data collection/depth/0/depth.zarr/"
                        ".zattrs]; typing a value always wins). The "
                        "paper default is 32 and the reference viewer uses 16, "
                        "but at --fstereo-scale 0.5 more iterations buy "
                        "nothing: on 9 real stereo frames the disagreement "
                        "with the reference setting (16 iters, scale 1.0) is "
                        "median 0.165 px at 8 iters vs 0.161 at 16, for "
                        "44.2 vs 58.8 ms eager [rov_gui/tools/"
                        "fstereo_bench_out/sweep.txt]. Scale is the knob "
                        "that costs accuracy; this one only costs time.")
    g.add_argument("--fstereo-scale", type=float, default=None,
                   help="downscale the pair before inference, <=1 (default: "
                        "0.5, or 1.0 with --policy — the training depth "
                        "store's setting; typing a value always wins). "
                        "THE frame-rate knob, and THE accuracy "
                        "knob. Against the reference setting on 9 real stereo "
                        "frames at 8 iters [rov_gui/tools/fstereo_bench_out/"
                        "sweep.txt]: at 0.5 the disparity differs by >1 px on "
                        "11.1%% of pixels (edges; median 0.17 px), at 0.75 on "
                        "7.0%%, at 1.0 on 1.8%%. Cost eager: 44 / 75 / 113 ms "
                        "per frame. 1 px at 1 m is ~3.7%% of range on this "
                        "baseline. Use 1.0 when edge accuracy matters more "
                        "than rate. With the CUDA graph (default) 0.5 replays "
                        "in 37 ms.")
    g.add_argument("--fstereo-size", default=None, metavar="WxH", type=_size_arg,
                   help="run the network at exactly WxH instead of "
                        "--fstereo-scale x the 640x400 mono size. Any aspect "
                        "ratio: disparity is a horizontal length, so only the "
                        "width ratio is undone. Buys little once the forward "
                        "is a CUDA graph — the cost is kernel COUNT, not "
                        "pixels. Graph replay on 9 real pairs, idle RTX 5090 "
                        "[rov_gui/tools/fstereo_bench_out/graph.txt]: 320x200 "
                        "36.8 ms, 224x224 30.5, 224x140 25.5, 160x100 23.5 "
                        "but unusable; against a reference-disagreement >1 px "
                        "on 11.1%%, 12.3%%, 17.0%%, 50.6%% of pixels "
                        "[sweep.txt]. 224x224 is the one worth having if 7 ms "
                        "matters more than a point of edge accuracy.")
    g.add_argument("--no-fstereo-graph", dest="fstereo_graph",
                   action="store_false", default=True,
                   help="run the network eagerly instead of replaying it as "
                        "one CUDA graph. The graph is the default because the "
                        "forward is ~6,500 kernel launches and launch-bound: "
                        "replay is 37 vs 45 ms on an idle GPU [rov_gui/tools/"
                        "fstereo_bench_out/graph.txt] and, being one call, "
                        "immune to the station's GIL contention that made "
                        "eager read 81 ms on the panel. Output is identical "
                        "(np.array_equal, 9 pairs x 10 settings). Capture failing "
                        "on a new torch/driver falls back to eager by itself "
                        "and says so in the log; this flag is for skipping "
                        "the attempt.")
    g.add_argument("--fstereo-fill", type=int, default=2, metavar="N",
                   help="passes of scatter-gap fill on the projected depth "
                        "(default: %(default)s; 0 = off). Projecting the "
                        "rectified-left map onto the colour grid writes one "
                        "destination pixel per source pixel, so wherever the "
                        "surface is slanted or steps in depth the pixel "
                        "between two samples is never written — a web of "
                        "1-3 px black curves over every depth gradient, "
                        "measured at 4.68%% of the grid on real C3 frames. "
                        "This closes them with a NEIGHBOURING SAMPLE'S "
                        "millimetres (grayscale dilation written only into "
                        "empty pixels), never an average: 2 passes reach "
                        "100.00%% fill for 0.14 ms and change zero measured "
                        "pixels [rov_gui/tools/fstereo_bench_out/"
                        "hardware_20260902_fill.txt]. The HUD keeps the two "
                        "apart — 'valid 97%%->95%%+5' means 95 measured, 5 "
                        "filled — so a repair is never read as a measurement.")
    g.add_argument("--fstereo-alpha", type=float, default=None, metavar="A",
                   help="cv2.stereoRectify alpha for the host rectification "
                        "(default: 0.0, or 0.5 with --policy; typing a value "
                        "always wins). 0 crops the rectified pair to its "
                        "valid region, which is what the learned matcher "
                        "wants; the policy's observation is a 400x400 centre "
                        "crop of the raw CAM_B 640x400 grid re-projected "
                        "from the rectified map, and at alpha 0 that crop is "
                        "not covered (<98.5%%, refused) while 0.5 covers "
                        "99.99%% of it with a 98%% valid pair [유도: "
                        "depth-parity review geo_check]. Recorded in the "
                        "run meta (rig.alpha).")
    g.add_argument("--land-dry-run", action="store_true",
                   help="RUN THE POLICY ON DRY LAND with the vehicle unable to "
                        "move (2026-09-03). There are no AprilTags on a bench, "
                        "so the localizer is replaced by a SYNTHETIC stationary "
                        "fix at the camera rate: the pose never changes and the "
                        "stamp always advances. That is not an approximation — "
                        "the policy's proprio is RELATIVE (inv(T_now) @ T_i), so "
                        "for a stationary vehicle it is identically zero/identity "
                        "whatever the absolute pose is, i.e. bit-identical to "
                        "perfect localization. Only the STAMP has to move, "
                        "because EtaHistory's degenerate test reads stamps, not "
                        "values. What this is for: watching what trajectory the "
                        "network proposes from a real depth image, with no way "
                        "to act on it.\n"
                        "FORCES, and cannot be overridden: no command sink "
                        "(--allow-command is ignored), no MAVLink rate requests, "
                        "and a SEPARATE run tree (data/*/*_landdry/) so a "
                        "bench record can never join or be pooled with a water "
                        "run. Relaxes the arm/flight-mode/telemetry engage gates "
                        "— they exist to protect a transmission that cannot "
                        "happen here — and the divergence guard, which would "
                        "otherwise stop the mission seconds in when the walking "
                        "reference leaves the frozen pose behind.\n"
                        "DO NOT ARM THE VEHICLE. The autopilot spins thrusters "
                        "on its own in any mode but MANUAL, with no byte from "
                        "this station; nothing here can prevent that.")
    g.add_argument("--fstereo-view", default="native",
                   choices=("native", "color"),
                   help="which map the depth PANEL draws (default: "
                        "%(default)s). native = the network's own rectified-"
                        "left 640x400 output, which is exactly what "
                        "FoundationStereo/UMI_Underwater/"
                        "oakd_foundation_stereo.py shows: ~100%% valid, full "
                        "resolution, no holes. color = the map projected onto "
                        "the colour grid, which is pixel-aligned with the RGB "
                        "panel but is a forward scatter through "
                        "--fstereo-align-size (400x250) and therefore loses "
                        "~10%% of the grid to occlusion and resampling and is "
                        "upscaled back. ONLY THE PICTURE AND ITS CURSOR "
                        "READOUT change: FoundationPose, the depth-vs-MAP "
                        "check and the policy's own input are unaffected "
                        "(the policy has always eaten the native map).")
    g.add_argument("--fstereo-palette", default="umi",
                   choices=("umi", "adaptive"),
                   help="WHICH COLOUR RULE the depth panel is painted with "
                        "(default: %(default)s). umi = the shared rule in "
                        "rov_gui/depth_colour.py: TURBO, warm = NEAR, FIXED "
                        "0.20-3.00 m, invalid black, and NO per-frame "
                        "adaptation — the same function that paints the land "
                        "FoundationStereo videos (umi_handheld/make_depth_"
                        "trajectory_video.py) and `depth_compare`, so red is "
                        "0.2 m and dark blue is 3.0 m on both screens and the "
                        "two can be compared by eye. adaptive = the picture "
                        "this panel drew from 2026-09-02: JET with a per-frame "
                        "2-98 percentile range and a per-frame 64-knot "
                        "histogram equalisation, which has more contrast on a "
                        "flat scene but gives the same distance a different "
                        "colour in every frame. AFFECTS ONLY THE PICTURE, ITS "
                        "COLOUR BAR and the ui_*.mp4 recording: the "
                        "millimetres the cursor reads, the policy's input, "
                        "FoundationPose and --record-depth are untouched.")
    g.add_argument("--fstereo-domain", default="obs", choices=("obs", "metric"),
                   help="spacing of the --fstereo-palette umi ramp (default: "
                        "%(default)s). obs = inverse depth, which is what the "
                        "policy's own channel 0 holds, so the panel is spaced "
                        "like the training observation; metric = linear in "
                        "metres. Match whatever the land picture beside it was "
                        "rendered with (`--domain` on the same two tools).")
    g.add_argument("--no-fstereo-equalize", dest="fstereo_equalize",
                   action="store_false",
                   help="ONLY UNDER --fstereo-palette adaptive. "
                        "Colourise the depth panel on a LINEAR ramp instead "
                        "of the reference viewer's histogram equalisation. "
                        "Equalised is the default because one far corner in "
                        "an otherwise close scene flattens a linear ramp "
                        "completely — the palette follows the pixel "
                        "distribution, not the millimetre span.")
    g.set_defaults(fstereo_equalize=True)
    g.add_argument("--no-fstereo-autorange", dest="fstereo_autorange",
                   action="store_false",
                   help="ONLY UNDER --fstereo-palette adaptive. "
                        "Colourise the depth PANEL over the fixed 0.3-6 m "
                        "scale instead of this frame's 2-98 percentile. The "
                        "auto range is on by default because a manipulation "
                        "scene lives inside ~1.5 m, i.e. the bottom fifth of "
                        "the fixed scale, and the panel goes uniformly dark — "
                        "which is what made it look broken next to "
                        "FoundationStereo/UMI_Underwater/oakd_foundation_"
                        "stereo.py, whose viewer has always auto-ranged. "
                        "AFFECTS ONLY THE PICTURE AND ITS COLOUR BAR: the "
                        "millimetres the cursor reads, the policy's input, "
                        "FoundationPose and every recording are untouched. "
                        "Turn it off to compare colours against an older "
                        "recording.")
    g.set_defaults(fstereo_autorange=True)
    g.add_argument("--fstereo-align-size", default="400x250", metavar="WxH",
                   help="grid the learned depth is projected onto (default: "
                        "%(default)s). THIS IS WHAT KEEPS THE PICTURE SOLID. "
                        "The rectified mono pair sees 82.6 deg while the "
                        "colour lens sees 63.9, so only 50.4%% of the mono "
                        "samples land inside the colour frame at all — a "
                        "640x360 grid can never exceed ~56%% fill and the rest "
                        "is black stripes. Measured fill: 640x360 50.6%%, "
                        "480x270 89.9%%, 400x250 100%%. Bigger is not sharper "
                        "here, it is holier.")

    g = p.add_argument_group(
        "object tracking (SAM2) — OFF unless --pose")
    g.add_argument("--pose", action="store_true",
                   help="build the object tracker. OFF by default and that is "
                        "deliberate: it loads torch and SAM2 (seconds to tens "
                        "of seconds, ~1.5 GB of VRAM) into the same process "
                        "that commands the vehicle, so it is opt-in. Without "
                        "it nothing is imported and the video worker does not "
                        "even copy a frame. Needs the rovgui-pose env — "
                        "'./c3 env' shows which interpreter the GUI resolved.")
    g.add_argument("--pose-src", default=None, metavar="DIR",
                   help="where the SAM2/FoundationPose project lives (default: "
                        "$SAM2_LIVE_ROOT, else ~/Desktop/New Folder). That tree "
                        "is READ-ONLY to this repo: rov_gui/perception/ adapts "
                        "it, never edits it.")
    g.add_argument("--pose-model", default="tiny",
                   choices=("tiny", "small", "base_plus", "large"),
                   help="SAM2 size (default: %(default)s). Measured on this "
                        "RTX 5090 by the upstream project: tiny 84.5 fps / 777 "
                        "MB, small 78.5, base_plus 60.6, large 37.1 fps / 1523 "
                        "MB. tiny is the default because the station is sharing "
                        "the GPU with nothing else that matters and 84 fps "
                        "against a 30 fps camera leaves headroom.")
    g.add_argument("--pose-mesh", default=None, metavar="MODEL.obj",
                   help="object mesh for 6-DoF pose, in METRES. With it, a "
                        "click registers in ~0.73 s and then tracks at 40-60 "
                        "Hz, and nothing is reconstructed. Without it the "
                        "station reconstructs one on site (see --pose-no-build)"
                        ". FoundationPose needs depth either way, so pose only "
                        "works at 0.3-0.8 m: past about 2 m the stereo noise "
                        "exceeds 50 mm and registration simply fails.")
    g.add_argument("--pose-no-build", action="store_true",
                   help="track only — never collect reference views and never "
                        "reconstruct. Gives a mask and no 6-DoF pose, which is "
                        "a useful mode on its own: it is the one with no "
                        "working-distance limit and no GPU-minutes. Without "
                        "this flag and without --pose-mesh, clicking an object "
                        "starts the model-free path (collect while you orbit "
                        "-> reconstruct ~2 min in a child process -> pose), "
                        "which is what the upstream tool does.")
    g.add_argument("--pose-ref-dir", default="data",
                   metavar="DIR",
                   help="ROOT of the dated tree for on-site reconstructions "
                        "(default: %(default)s). Each attempt gets its own "
                        "<DIR>/YYYYMMDD/MMDD_HHMMSS_obj/ holding the reference views, the "
                        "training log and model/model.obj — pass that .obj to "
                        "--pose-mesh next time to skip the two minutes. The "
                        "path must not contain 'rgb': the reconstruction "
                        "derives sibling paths by replacing that substring.")
    g.add_argument("--pose-max-arc", type=float, default=75.0, metavar="DEG",
                   help="stop collecting after this much accumulated rotation "
                        "(default: %(default)s). Measured upstream on "
                        "synthetic data: the per-view pose error is 9.7 mm at "
                        "80 deg, 87 mm at 100 and 283 mm at 120 — it does not "
                        "degrade, it falls off a cliff. Raising this does not "
                        "buy more coverage, it buys a wrong mesh.")
    g.add_argument("--depth-scale", type=float, default=0.64, metavar="K",
                   help="multiply the C3 depth stream's millimetres by this, "
                        "at the source, before ANY consumer (panel probe, "
                        "reference capture, FoundationPose, depth-vs-MAP). "
                        "Interim correction for the stereo depth reading LONG "
                        "in water: measured 2026-08-23, mesh 187 mm from a "
                        "caliper-measured 119.73 mm object (1.56x) and "
                        "depth-vs-MAP 1.71x over the mat, while colour PnP "
                        "was mm-accurate. ON BY DEFAULT (%(default)s = 1/1.56) "
                        "since 2026-08-24: it used to default to 1.0 and "
                        "forgetting it was silent — the 10:11 and 10:19 runs "
                        "that day placed the object 1.55-1.57x down the camera "
                        "ray, two tag rows past the object and half a metre "
                        "UNDER the mat, with nothing in the log saying the "
                        "correction was off. Read depth-vs-MAP to confirm: "
                        "~1.0x means it holds. Pass 1.0 to disable — which is "
                        "what an IN-AIR bench test wants, since this factor is "
                        "the in-water one. NOTE a mesh bakes in whatever scale "
                        "was in force when its reference views were captured, "
                        "so changing this invalidates existing meshes: turn it "
                        "on FIRST, then re-capture, then fly. The real fix is "
                        "an in-water stereo recalibration.")
    g.add_argument("--pose-object-size", type=float, default=None,
                   metavar="MM",
                   help="the object's LONGEST dimension, measured with a "
                        "tape. Used to CHECK a reconstruction the moment it is "
                        "built, never to rescale it: this path reconstructs "
                        "from metric RGB-D, so the mesh already has a size and "
                        "scaling it uniformly would shrink the dimensions that "
                        "are right to fix the one that is wrong. Eight "
                        "reconstructions of one object on 2026-08-23 kept the "
                        "same ~100 x 170 mm cross-section and grew only in "
                        "LENGTH, 171 -> 470 mm, in step with the vertex count "
                        "— a mask swallowing floor and shadow, not a scale "
                        "error. The only symptom downstream is FoundationPose "
                        "failing to hold a pose hours later, so this catches "
                        "it at the build. OPTIONAL since 2026-08-24: without "
                        "it the same failure is caught automatically against "
                        "the largest silhouette the reference views observed "
                        "(a tape-free metric bound), and a mesh past either "
                        "limit is REFUSED, not just warned about. Give the "
                        "tape number when you have it — it is the stricter "
                        "check. Ignored with --pose-mesh.")
    g.add_argument("--pose-max-views", type=int, default=20, metavar="N",
                   help="reference-view budget (default: %(default)s). Under "
                        "6 the station refuses to reconstruct; 10+ is "
                        "recommended; the reference implementation uses 16.")
    g.add_argument("--pose-lost-timeout", type=float, default=15.0,
                   metavar="S",
                   help="how long the tracker keeps trying to REACQUIRE an "
                        "object it lost before giving up and asking for "
                        "another click (default: %(default)s). Underwater, "
                        "motion blur and a moving hull lose the mask far more "
                        "often than a bench does — raise this if the run is "
                        "ending in 'could not reacquire' rather than in a "
                        "finished reconstruction. Costs nothing but patience: "
                        "while it is reacquiring nothing is recorded and "
                        "nothing is thrown away.")
    g.add_argument("--pose-lost-grace", type=float, default=3.0, metavar="S",
                   help="how long a lost mask is tolerated before the overlay "
                        "calls it lost at all (default: %(default)s). Raising "
                        "it hides brief dropouts instead of surviving long "
                        "ones — --pose-lost-timeout is the knob for that.")
    g.add_argument("--pose-box3d", action="store_true",
                   help="draw the full 12-edge oriented box instead of the "
                        "three axes only")
    g.add_argument("--pose-ckpt", default=None, metavar="PATH",
                   help="explicit SAM2 checkpoint; by default the upstream "
                        "search path finds it ($SAM2_CHECKPOINT_DIR first)")

    g = p.add_argument_group(
        "closed-loop MPC (AprilTag + acados) — OFF unless --mpc")
    g.add_argument("--mpc", action="store_true",
                   help="build the closed-loop stack: AprilTag PnP off the C3 "
                        "colour stream -> the sim's EAOB+NMPC (verbatim from "
                        "bluerov2_mujoco_marinegym/dobmpc, acados SQP-RTI, "
                        "20 Hz) -> MANUAL_CONTROL through the normal command "
                        "sink and all of its gates. OFF by default: it "
                        "imports casadi+acados and code-generates the solver "
                        "at startup (measured 1.3 s build, 2.1 ms probe in "
                        "rovgui-pose — rov_gui/control/smoke.py). Engaging "
                        "additionally needs COMMAND ENABLE, an armed vehicle "
                        "in MANUAL, and a fresh tag fix. In --source demo it "
                        "closes the loop against the synthetic plant.")
    g.add_argument("--nav-config", default="config/hw_nav.yaml",
                   help="AprilTag navigation config (default: %(default)s)")
    g.add_argument("--mpc-config", default="config/hw_mpc.yaml",
                   help="controller/mission config (default: %(default)s)")
    g.add_argument("--nav-geometry", default=None, choices=("wall", "floor"),
                   help="override the config's geometry: 'wall' = one tag on "
                        "the pool wall, forward-level camera, crab square; "
                        "'floor' = the surveyed floor map, camera remounted "
                        "DOWN (needs a re-measured extrinsic in hw_nav.yaml)")
    g.add_argument("--mpc-mode", default=None,
                   choices=("none", "mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned",
                            "pid", "rl"),
                   help="override hw_mpc.yaml mode — the LOW level (the "
                        "follower) the trajectory panel starts on; the panel "
                        "can change it between engagements. One solver serves "
                        "all four MPC names: the 'dob' prefix adds the EAOB "
                        "disturbance feedforward, the '_tuned' suffix splits "
                        "the position cost into along-track (cheap) and "
                        "cross-track (expensive) in the path frame "
                        "(hw_mpc.yaml mpc_tuned:). 'none' = TELEOP: the "
                        "station commands nothing, you fly on the joystick; "
                        "a Diffusion Policy mission still infers and draws "
                        "(2026-09-11; --policy-observe is the launch alias).")
    g.add_argument("--rov-model", default=None,
                   choices=("heavy", "heavy_c3", "heavy_gripper"),
                   help="override hw_mpc.yaml rov_model (which dobmpc plant "
                        "parameters the controller carries)")
    g.add_argument("--replay-session", default=None, metavar="DIR",
                   help="override hw_mpc.yaml replay.session: the demo folder "
                        "the `replay` mission shape re-flies (must hold "
                        "poses.npy from `python -m umi_handheld."
                        "extract_pose`). Speed cap, blend and the gripper "
                        "switch stay in the replay: block.")

    g = p.add_argument_group(
        "diffusion policy — OFF unless --policy")
    g.add_argument("--policy", action="store_true",
                   help="build the diffusion-policy worker (backends/policy.py): "
                        "the trained UMI DiffusionTransformerTimmPolicy "
                        "loads onto the GPU at startup and, while a `policy` "
                        "mission is running under --mpc, turns the depth "
                        "stream + the tag pose into 1 s action chunks the "
                        "NMPC tracks through the replay plan seam. The "
                        "CHECKPOINT is hw_mpc.yaml policy.ckpt at startup and "
                        "is CHOSEN IN THE PANEL: HIGH level = Diffusion "
                        "Policy shows a ckpt picker; picking a file reloads "
                        "the network on the GPU (refused while a policy "
                        "mission runs). There is no --policy-ckpt flag "
                        "(removed 2026-09-11, operator request). OFF by "
                        "default: torch + a 6 GB checkpoint in the process "
                        "that commands the vehicle. Needs --mpc to have any "
                        "consumer and --fstereo for the training-parity depth "
                        "(see --policy-allow-device-depth). "
                        "HARDWARE-UNVERIFIED: the first pool run is a "
                        "pipeline smoke, not a grasp test (rov_gui/README.md).")
    g.add_argument("--policy-repo", default=None, metavar="DIR",
                   help="override hw_mpc.yaml policy.repo (the UMI checkout "
                        "whose diffusion_policy package built the checkpoint)")
    g.add_argument("--policy-allow-device-depth", action="store_true",
                   help="let the policy consume the C3's OWN depth stream "
                        "(colour-aligned, x--depth-scale) when --fstereo is "
                        "not given. REFUSED by default: that depth covers "
                        "~64%% of the policy's field of view [유도] and comes "
                        "from a different matcher than the training data; "
                        "--fstereo is the parity path. Bench experiments "
                        "only.")
    g.add_argument("--policy-observe", action="store_true",
                   help="start with LOW level = None (teleop) preselected — "
                        "identical to picking None in the trajectory panel; "
                        "kept for scripts (2026-09-11: the runtime choice "
                        "replaced the launch-only mode). Under LOW None the "
                        "DP infers, its chunks are composed into datum-NED "
                        "plans, filtered, stitched and DRAWN on the "
                        "trajectory panel; ctrl.step() is never called and "
                        "not one axis leaves this station, so YOU fly on the "
                        "joystick and check whether the reference it draws "
                        "makes sense. Disarms the five automatic stops that "
                        "assume a follower (divergence guard, escalation "
                        "latch, bridge-too-long, workspace box, tag-loss "
                        "disengage), forces the jaw off, and uses "
                        "policy.observe_max_run_s. Needs --policy (and "
                        "--mpc, which owns the mission); disagrees with any "
                        "other --mpc-mode (refused). RECORD BOUNDARY: runs "
                        "land in data/*/*_observe/, every CSV row "
                        "carries observe=1 and nothing measured here is "
                        "comparable with a closed-loop run.")
    g.add_argument("--policy-weights", default=None,
                   choices=("ema_model", "model"),
                   help="which state dict inside the checkpoint to fly "
                        "(default: ema_model, what selected.json names). On "
                        "the 2026-09-01 checkpoint `model` is measurably "
                        "BETTER on held-out actions — 0.000842 vs 0.001535 "
                        "action MSE, 20.5 vs 30.7 mm position RMSE over the "
                        "full validation split at DDIM 8 [측정 2026-09-06: "
                        "episodes 6/32/48/56 by get_val_mask(75, 0.05, 42), "
                        "219 windows]. The cause is not EMA itself: "
                        "EMAModel.step averages only named_parameters(), so "
                        "BatchNorm's running statistics were never averaged "
                        "and the EMA copy still carries the timm ImageNet "
                        "ones; swapping just those 108 buffers recovers 92%% "
                        "of the gap. RECORD BOUNDARY: the choice is in the "
                        "run meta and runs on different weights must not be "
                        "pooled.")
    g.add_argument("--record-depth", action="store_true",
                   help="WRITE THE UNDERWATER DEPTH TO DISK. Off by default, "
                        "and until it is given a run persists NO depth at "
                        "all: not the 224x224 observation the policy "
                        "consumed, not the uint16 millimetre map, and not "
                        "the stereo pairs either — so the depth cannot be "
                        "inspected afterwards and cannot be regenerated "
                        "offline. (The ui_*.mp4 is a recording of the "
                        "COLOURISED, per-frame auto-ranged, "
                        "histogram-equalised, downscaled screen. It is a "
                        "picture of a picture, not depth.) With this flag "
                        "each depth frame the policy actually used is "
                        "written losslessly to "
                        "<run>/policy_obs/{obs,depth}/NNNNNN.png with an "
                        "index.csv and a meta.json carrying the obs recipe. "
                        "Render it against the land training obs on ONE "
                        "colour rule with `python -m "
                        "rov_gui.tools.depth_compare pair --run <run>`. The "
                        "writer is a background thread behind a bounded "
                        "queue that DROPS frames rather than ever blocking "
                        "the control path; drops are counted in meta.json.")
    g.add_argument("--record-depth-max", type=int, default=None, metavar="N",
                   help="stop after N recorded frames (default: 4000). A disk "
                        "budget, so a flag left on cannot fill the disk "
                        "mid-dive; reaching it is not an error and the run "
                        "continues unaffected. Frames are fed at roughly "
                        "2 per inference, so ~4/s at period_s 0.5 — call it "
                        "~16 min and ~2 GB [예측: not measured; replace both "
                        "with `du -sh` on the first recorded run]. There is "
                        "deliberately no every-Nth option: it would "
                        "phase-lock onto one member of the observation pair "
                        "and hand back a biased half of the frames.")
    g.add_argument("--record-depth-rgb", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="with --record-depth, also file the C3 COLOUR frame "
                        "that was current when each observation was built "
                        "(default: %(default)s) — <run>/policy_obs/rgb/*.jpg, "
                        "with `rgb_dt_ms` in index.csv saying how far apart "
                        "the pair was (colour runs ~30 Hz, the recorded depth "
                        "~5 Hz, so near-simultaneous, never simultaneous). "
                        "JPEG, not PNG: the obs and the millimetre map are "
                        "measurements and stay lossless, this is a picture the "
                        "policy never sees. It is what lights up the C3 RGB "
                        "column of rov_gui/tools/export_run_html.py.")
    g.add_argument("--record-depth-rgb-quality", type=int, default=85,
                   metavar="Q", help="JPEG quality for those colour frames "
                                     "(default: %(default)s).")
    g.add_argument("--record-stereo", action="store_true",
                   help="WRITE THE RAW MONO PAIRS TO DISK — FoundationStereo's "
                        "INPUT, so this run's depth can be COMPUTED AGAIN "
                        "later at other settings. --record-depth saves the "
                        "network's OUTPUT (the observation and the millimetre "
                        "map); no setting can be changed from an output, so "
                        "until this flag a run was welded to the iters / scale "
                        "/ checkpoint it flew with. Writes RAW (unrectified) "
                        "left+right lossless PNG to <run>/stereo/ with an "
                        "index.csv and a meta.json carrying the rig "
                        "(rectification geometry + calibration provenance) and "
                        "the FoundationStereo settings this run used, so both "
                        "the flown depth and a re-rectified one are "
                        "reproducible. RAW, not rectified, because "
                        "rectification is a function of the calibration and "
                        "baking today's in would destroy the ability to apply "
                        "the underwater re-calibration later. SIZE: a 640x400 "
                        "pair is 296 KiB lossless [측정 2026-09-06, 10 real "
                        "pairs through this recorder] = ~2.1 MB/s at the "
                        "measured 7.26 Hz, so see --record-stereo-max. "
                        "Needs --fstereo (no learned-depth worker, no pairs) "
                        "and --mpc (the run folder).")
    g.add_argument("--record-stereo-max", type=int, default=None, metavar="N",
                   help="stop after N recorded PAIRS (default: 6000 ~ 1.7 GB "
                        "and ~14 min at 7.26 Hz [유도 from the measured 296 KiB "
                        "pair]). A disk budget, so a flag left on cannot fill "
                        "the disk mid-dive; reaching it is not an error and "
                        "the run continues unaffected.")
    g.add_argument("--record-stereo-every", type=float, default=0.0,
                   metavar="SEC",
                   help="record at most one pair every SEC seconds (default: "
                        "%(default)s = every pair the worker took). SAMPLING "
                        "COSTS SOMETHING SPECIFIC: the policy's observation is "
                        "built from a PAIR OF CONSECUTIVE depth frames, so a "
                        "sampled recording can no longer reconstruct the "
                        "observations the policy actually consumed — only "
                        "individual frames. Sample for a long survey; keep "
                        "every pair for anything that will be compared against "
                        "policy_obs/.")
    g.add_argument("--record-depth-obs-only", action="store_true",
                   help="write only the 224x224 observation, not the "
                        "640x400 millimetre map. Roughly a quarter of the "
                        "bytes. The obs is what the land/water comparison "
                        "actually needs; the millimetre map is for checking "
                        "absolute distances against a tape.")
    g.add_argument("--policy-allow-fs-mismatch", action="store_true",
                   help="fly the policy on FoundationStereo settings that "
                        "differ from the training depth store's (iters 16, "
                        "scale 1.0, the same checkpoint). Refused by default "
                        "because the policy never saw depth from another "
                        "setting; the mismatch is recorded in the run meta "
                        "either way.")

    g = p.add_argument_group(
        "live ORB-SLAM3 pose — OFF unless --slam")
    g.add_argument("--slam", action="store_true",
                   help="take the vehicle pose from a live stereo ORB-SLAM3 "
                        "instead of AprilTag PnP, for a run in a room with no "
                        "tags in it — a HANDHELD land test of the diffusion "
                        "policy. The station stays the only owner of the C3 "
                        "and tees the raw mono pair it already pulls for "
                        "--fstereo into a child `oakd_live_slam` process, "
                        "whose Pangolin window IS the map; the policy's "
                        "predicted knots are sent back and drawn in it. "
                        "Implies the raw mono streams even without --fstereo. "
                        "HARDWARE-UNVERIFIED.")
    g.add_argument("--no-slam-viewer", dest="slam_viewer",
                   action="store_false",
                   help="run ORB-SLAM3 headless (--no-viewer). The map window "
                        "is the whole point of --slam, so this is for a "
                        "recording-only run or a machine with no X display.")
    g.add_argument("--no-slam-clahe", dest="slam_clahe", action="store_false",
                   help="do NOT equalise the SLAM images. Measured on "
                        "recorded episode 0, CLAHE (clip 3.0, 8x8) took the "
                        "ORB keypoint median from 384 to 1243 and the "
                        "fraction of frames clearing ORB-SLAM3's "
                        "500-keypoint initialisation gate from 26%% to 100%% "
                        "[측정: 2026-09-03 feed_episode.py --check]; the "
                        "training SLAM that produced the demonstration pose "
                        "labels also ran with it. Expect no map without it.")
    g.add_argument("--slam-attitude", action="store_true",
                   help="publish roll/pitch/yaw from the SLAM solution as "
                        "VehicleImu. OFF by default because ArduSub's own "
                        "ATTITUDE is the better source and bus.vehicle_imu is "
                        "last-writer-wins — turn this on only when the "
                        "vehicle is unpowered.")
    g.add_argument("--slam-features", type=int, default=1200,
                   help="ORBextractor.nFeatures (default: %(default)s)")
    g.add_argument("--slam-ini-fast", type=int, default=20,
                   help="ORBextractor.iniThFAST (default: %(default)s). Lower "
                        "accepts weaker corners — the right trade when the "
                        "alternative is no map at all.")
    g.add_argument("--slam-min-fast", type=int, default=7,
                   help="ORBextractor.minThFAST (default: %(default)s)")

    g = p.add_argument_group("ui")
    g.add_argument("--ui-fps", type=float, default=60.0,
                   help="repaint rate (default: %(default)s). This is real "
                        "display latency: at 30 Hz a frame waits up to 33 ms "
                        "for the next paint, on top of the camera's own 34 ms.")
    g.add_argument("--fullscreen", action="store_true")
    g.add_argument("--rec-dir", default="data",
                   help="ROOT of the dated run tree (default: %(default)s). "
                        "Everything one run produces — the UI recording, the "
                        "feed recordings, the nav recording, the controller "
                        "CSV and events.log — lands together in "
                        "<rec-dir>/YYYYMMDD/MMDD_HHMMSS[_kind]/ (kind: "
                        "_observe / _landdry / _dryrun / _obj). See "
                        "rov_gui/runstore.py.")
    g.add_argument("--rec-fps", type=float, default=12.0,
                   help="UI recording frame rate (default: %(default)s)")
    g.add_argument("--demo-object", default="still",
                   choices=("still", "drift", "orbit"),
                   help="--source demo --pose --mpc only: how the SYNTHETIC "
                        "tracked object behaves in the pool frame. still = "
                        "sits there (arming a follow must not move the "
                        "vehicle); drift = translates slowly (the "
                        "feedforward case); orbit = rotates in place (the "
                        "orbit case, and the one a wrong object_nav.yaw_axis "
                        "shows up in fastest). Default: %(default)s.")
    g.add_argument("--demo-leak-after", type=float, default=None,
                   metavar="S",
                   help="--source demo only: fake a flooding enclosure S "
                        "seconds in, so the LEAK banner and sensor row can be "
                        "seen before anyone has to trust them. Has no effect "
                        "on hardware, where leak comes from ArduSub's "
                        "\"Leak Detected\" STATUSTEXT and nothing else.")

    g = p.add_argument_group("preflight (hardware source only)")
    g.add_argument("--no-preflight", action="store_true",
                   help="skip the readiness checks and go straight to connecting")
    g.add_argument("--preflight-only", action="store_true",
                   help="run the readiness checks and exit, without connecting")
    g.add_argument("--preflight-strict", action="store_true",
                   help="treat warnings as failures")
    g.add_argument("--preflight-timeout", type=float, default=3.0,
                   help="per-check network timeout in seconds (default: %(default)s)")
    return p


#: Defaults that depend on ANOTHER flag, and therefore cannot be argparse
#: defaults. Each entry is (attribute, value without --fstereo, value with it).
#: Kept as data so the test that checks "--fstereo changes the link budget"
#: reads the same table the program does.
FSTEREO_DEPENDENT_DEFAULTS = (
    # THE FRAME RATE. --fstereo replaces the depth path for the whole run, so
    # the device's own depth stream has no consumer left and its bandwidth
    # becomes the mono pair's frame rate. The pair is 2.05 Mbit/s per camera
    # per fps and cannot be encoded (the device's VideoEncoder is never wired
    # to the mono cameras). Measured totals against the ~90 Mbit/s C3 link:
    #   colour + pair @12            =  59.0 Mbit/s ( 66%)
    #   colour + pair @15            =  71.3        ( 79%)  <- shipped
    #   colour + pair @18            =  83.6        ( 93%, tight)
    #   colour + pair @20            =  91.8        (102%, OVER)
    # 15 fps is the fastest with headroom for a busy scene. The network is no
    # longer the slower side: 45 ms end-to-end as a CUDA graph at scale 0.5
    # [측정 2026-09-02, RTX 5090 idle, real 640x400 pairs:
    # rov_gui/tools/fstereo_bench_out/session.txt], so the pair's frame rate
    # is now the cap, and the link is what holds it at 15.
    ("depth_fps", 20.0, 15.0),
)


#: Defaults that depend on --policy. THE SPEED/PARITY TRADE, chosen for speed
#: (2026-09-02, operator: "최소 10 fps"). Spec v2 A16/A17 originally put these
#: on the TRAINING depth store's settings — iters 16, scale 1.0 [측정:
#: ~/Desktop/data collection/depth/0/depth.zarr/.zattrs] — which measured
#: 7.26 Hz / solve 137 ms / panel latency 243 ms [측정: rov_gui/tools/
#: fstereo_bench_out/policy_bench_20260902_141224.json, C3 실기, 330 frames].
#:
#: That bought instrument parity and paid for it in TIME parity: at 7.26 Hz the
#: two depth frames of one observation are 138 ms apart, 2.07x the 66.7 ms
#: stride the network trained on (KNOWN_ISSUES 2026-09-02, which says outright
#: that which mismatch is cheaper is unknown before a hardware A/B). So the
#: run was already off-distribution in one axis while paying full price for
#: the other.
#:
#: iters 8 / scale 0.75 measures 74.9 ms eager [측정: rov_gui/tools/
#: fstereo_bench_out/sweep.txt] = ~13 Hz, and is the CLOSEST row in that sweep
#: to the reference: median disagreement 0.099 px, p90 0.747 (the previously
#: shipped scale 0.5 is 0.165 / 1.095). Alpha stays 0.5 — that one is not a
#: speed knob, it is what covers the policy's 400x400 centre crop.
#: Each entry is (attribute, value without --policy, value with it); applied
#: only when the attribute is None, so a typed value always wins.
POLICY_DEPENDENT_DEFAULTS = (
    ("fstereo_iters", 8, 8),
    ("fstereo_scale", 0.5, 0.75),
    ("fstereo_alpha", 0.0, 0.5),
)


#: What the --policy defaults buy and cost, said at launch.
FS_RATE_NOTE = (
    "FoundationStereo at the --policy defaults (iters 8, scale 0.75) measures "
    "74.9 ms eager [측정: rov_gui/tools/fstereo_bench_out/sweep.txt, 9 real "
    "pairs, RTX 5090 idle] = ~13 Hz, against 140.2 ms = ~7 Hz at the training "
    "store's iters 16 / scale 1.0. This is DELIBERATELY off the training "
    "instrument (median disagreement 0.099 px, p90 0.747 vs that reference) "
    "and buys back the time axis: the depth pair is ~77 ms apart, near the "
    "66.7 ms training stride, instead of 138 ms = 2.07x it. Both settings are "
    "recorded in the run meta and obs_pair_dt_s is on every plan — which "
    "mismatch costs more is a hardware A/B nobody has run "
    "(KNOWN_ISSUES 2026-09-02). --fstereo-scale 1.0 --fstereo-iters 16 "
    "restores exact instrument parity at ~7 Hz.")


def resolve_defaults(a):
    """Fill in the defaults that depend on --fstereo / --policy. Idempotent.

    Argparse cannot express "default 20 unless another flag is set", and the
    obvious workaround — reading the value at each use site — spreads the rule
    over three files and lets them disagree. So the sentinel is None and it is
    resolved here, once, before anything reads the namespace.
    """
    on = bool(getattr(a, "fstereo", False))
    for name, off_value, on_value in FSTEREO_DEPENDENT_DEFAULTS:
        if getattr(a, name, None) is None:
            setattr(a, name, on_value if on else off_value)
    pol = bool(getattr(a, "policy", False))
    for name, off_value, on_value in POLICY_DEPENDENT_DEFAULTS:
        if getattr(a, name, None) is None:
            setattr(a, name, on_value if pol else off_value)
    # --policy-observe is the launch ALIAS of LOW level = None (2026-09-11,
    # operator request: the observe mode became the panel's runtime choice).
    # ONE source of truth downstream: MpcWorker reads `mpc_mode == "none"`
    # (its setup normalises the legacy `opts.policy_observe` too, for the
    # fixtures that set it directly), and the window seeds the LOW combo from
    # `opts.mpc_mode`. A typed, DIFFERENT --mpc-mode is a contradiction that
    # check_policy refuses rather than something to resolve silently.
    if bool(getattr(a, "policy_observe", False)) \
            and getattr(a, "mpc_mode", None) is None:
        a.mpc_mode = "none"
    # --land-dry-run OVERRIDES rather than defaults: these three are the whole
    # reason the mode may relax the arm and flight-mode engage gates, so a
    # typed --allow-command must not be able to take them back. Done here so
    # every reader of the namespace — backend, workers, meta — sees one answer.
    if bool(getattr(a, "land_dry_run", False)):
        a.allow_command = False       # no sink is built at all (hardware.py)
        a.mavlink_set_rates = False   # the one transmit outside the sink
    return a


def check_slam(a):
    """What ``--slam`` may and may not be combined with. At launch.

    Returns ``None`` or ``(level, message)``; ``main()`` prints and returns 2
    on ``"refuse"``. Deliberately NOT inside :func:`check_policy`, which
    returns early when ``--policy`` is off — the hazard below is about who may
    transmit thrust and has nothing to do with the policy being loaded.
    """
    if not bool(getattr(a, "slam", False)):
        return None
    if bool(getattr(a, "allow_command", False)):
        # --slam exists for ONE thing: a run where a human is CARRYING the
        # vehicle. --allow-command builds a MavlinkCommandSink that transmits
        # manual_control (hardware.py), and the station's E-STOP cannot stop a
        # vehicle it is not the only commander of — under NullCommandSink it
        # transmits nothing at all, which is the same property that makes the
        # mode safe and makes the button useless. Refuse here, where it costs
        # a second.
        return ("refuse",
                "--slam and --allow-command are mutually exclusive. --slam is "
                "the handheld land rig, where a person is holding the "
                "vehicle; --allow-command transmits thrust. Physical stop for "
                "a land run = the battery or tether switch, not the GUI.")
    if getattr(a, "source", "demo") != "hw":
        return ("refuse", f"--slam needs --source hw (got "
                          f"{getattr(a, 'source', 'demo')!r}): the bridge "
                          f"feeds ORB-SLAM3 the C3's raw mono pair, and no "
                          f"other source produces one.")
    if not bool(getattr(a, "slam_clahe", True)):
        return ("warn",
                "--no-slam-clahe: on recorded episode 0 that took the ORB "
                "keypoint median from 1243 to 384 and the fraction of frames "
                "clearing ORB-SLAM3's 500-keypoint gate from 100% to 26% "
                "[측정: 2026-09-03 feed_episode.py --check]. Expect no map.")
    return None


def check_policy(a):
    """What --policy can and cannot do with the rest of the flags. At launch.

    Returns ``None`` or ``(level, message)`` with ``level`` in
    ``("warn", "refuse")`` — unlike :func:`check_fstereo`, some of these are
    REFUSALS (``main()`` prints and returns 2), because flying the policy on
    the wrong depth instrument is not a rehearsal, it is a wrong measurement
    dressed as a run (spec v2 A16/A17). REFUSALS are evaluated first — a
    missing ``--mpc`` used to mask the missing-``--fstereo`` refusal behind a
    warning (verify 2026-09-02) — then every WARNING that applies is joined
    into one line:

    * ``--source hw --policy`` without ``--fstereo`` and without
      ``--policy-allow-device-depth``: REFUSED — the device's colour-aligned
      depth covers only part of the policy FOV [유도: CAM_A-aligned 63.7 deg
      inside the CAM_B 85.6 deg FOV] and is the x0.64-scaled on-device
      matcher; the training-parity depth is FoundationStereo.
    * ``--policy --fstereo`` whose FS settings differ from the training depth
      store's (read torch-free from the hydra config beside the checkpoint
      and the dataset zip's ``.zattrs``; the EFFECTIVE FS checkpoint is
      compared, typed or default): REFUSED unless
      ``--policy-allow-fs-mismatch``; a missing config/zip is reported, not
      refused (the meta records what was and was not checked).
    * ``--policy`` without ``--mpc``: WARNING — the worker loads and infers
      for nobody (no consumer builds a `policy` mission).
    * ``--policy-allow-device-depth``: WARNING — a bench experiment, recorded
      in the meta (``depth_scale_applied`` 0.64, coverage).
    * ``--fstereo-alpha`` below 0.5 under ``--policy``: WARNING — the 224 obs
      crop is not covered at alpha 0 (97.92 % < 98.5 %, refused by the
      builder at run time; the mission would arm and never receive a plan).
    Also says so when the --policy defaults moved the FS settings, with the
    measured rate (FS_RATE_NOTE).
    """
    if bool(getattr(a, "policy_observe", False)):
        # A REFUSAL, not a warning: --policy-observe alone builds no policy
        # worker, so the station would come up looking like a DP monitor and
        # simply never draw anything — and it would ALSO have muted the
        # controller, i.e. an operator could press START on a square and
        # watch nothing happen with no gate having said why.
        if not bool(getattr(a, "policy", False)):
            return ("refuse", "--policy-observe needs --policy (it is a mode "
                              "for the diffusion-policy mission; without it "
                              "no DP worker is built and nothing would be "
                              "drawn — while the controller would still be "
                              "muted for every other shape).")
        if not bool(getattr(a, "mpc", False)):
            return ("refuse", "--policy-observe needs --mpc: the MpcWorker is "
                              "what arms the policy mission, composes the "
                              "plans and draws them. Without it the DP "
                              "worker infers for nobody.")
        _mm = getattr(a, "mpc_mode", None)
        if _mm is not None and str(_mm).lower() != "none":
            # The alias means LOW level None; a typed controller says the
            # opposite. Refuse instead of picking one: whichever won, the
            # operator would launch believing the other (2026-09-11).
            return ("refuse", f"--policy-observe and --mpc-mode {_mm} "
                              f"disagree: --policy-observe is the launch "
                              f"alias of LOW level = None (teleop; nothing "
                              f"is commanded), --mpc-mode {_mm} asks for a "
                              f"follower. Drop one of them — LOW level can "
                              f"be changed in the trajectory panel between "
                              f"engagements.")
    if bool(getattr(a, "record_depth", False)) and not bool(getattr(a, "mpc",
                                                                    False)):
        # The recorder files its frames in the CONTROLLER's run folder, which
        # is the only thing that knows the run TREE (a land dry-run and an
        # observe run get their own — MpcWorker._run_tree calls that a safety
        # property of the record) and the PIN (fixed at engagement, so the
        # depth joins the folder the CSVs are already in). No --mpc, no
        # controller, no correct answer — and the backend then refuses to
        # build a recorder rather than inventing a folder.
        return ("refuse", "--record-depth needs --mpc: the recording is filed "
                          "in the controller's run folder, and without the "
                          "MpcWorker there is nothing that knows which tree "
                          "(water / land_dryruns / policy_observe) or which "
                          "folder this run is. Add --mpc, or drop "
                          "--record-depth.")
    _rdm = getattr(a, "record_depth_max", None)
    if _rdm is not None and int(_rdm) <= 0:
        return ("refuse", "--record-depth-max must be positive; there is no "
                          "'unlimited'. It is a disk guard, and 0 would read "
                          "as one either way round.")
    if bool(getattr(a, "record_depth", False)) and not bool(getattr(a, "policy",
                                                                    False)):
        # A REFUSAL, not a warning. The recorder lives in the PolicyWorker, so
        # without --policy the flag is a silent no-op — the operator flies a
        # whole dive believing the depth is being written and surfaces with
        # nothing. That is the exact failure --record-depth exists to end, so
        # it must not be possible to ask for it and not get it.
        return ("refuse", "--record-depth needs --policy: the recorder is fed "
                          "by the diffusion-policy worker's observation "
                          "builder, so without --policy nothing would be "
                          "written and you would not find out until the dive "
                          "was over. Add --policy (with --fstereo), or drop "
                          "--record-depth.")
    if bool(getattr(a, "record_stereo", False)) and not bool(
            getattr(a, "fstereo", False)):
        # A REFUSAL for the same reason --record-depth's is: the pairs exist
        # only inside the learned-depth worker's mailbox, so without --fstereo
        # the flag is a silent no-op and the operator finds out after the dive.
        return ("refuse", "--record-stereo needs --fstereo: the raw mono pairs "
                          "are fed to the recorder by the learned-depth "
                          "worker, and without it nothing would be written. "
                          "Add --fstereo, or drop --record-stereo.")
    if bool(getattr(a, "record_stereo", False)) and not bool(
            getattr(a, "mpc", False)):
        return ("refuse", "--record-stereo needs --mpc: the pairs are filed in "
                          "the controller's run folder, and without the "
                          "MpcWorker there is nothing that knows which tree "
                          "(water / land_dryruns / policy_observe) or which "
                          "folder this run is. Add --mpc, or drop "
                          "--record-stereo.")
    _rq = getattr(a, "record_depth_rgb_quality", 85)
    if not (1 <= int(_rq) <= 100):
        return ("refuse", "--record-depth-rgb-quality must be 1..100.")
    _rsm = getattr(a, "record_stereo_max", None)
    if _rsm is not None and int(_rsm) <= 0:
        return ("refuse", "--record-stereo-max must be positive; there is no "
                          "'unlimited'. It is a disk guard, and 0 would read "
                          "as one either way round.")
    if float(getattr(a, "record_stereo_every", 0.0) or 0.0) < 0.0:
        return ("refuse", "--record-stereo-every cannot be negative; 0 means "
                          "every pair.")
    if not bool(getattr(a, "policy", False)):
        return None
    src = getattr(a, "source", "demo")
    fs = bool(getattr(a, "fstereo", False))
    allow_dev = bool(getattr(a, "policy_allow_device_depth", False))
    warns: list[str] = []
    # ---- refusals first
    if src == "hw" and not fs and not allow_dev:
        return ("refuse", "policy needs --fstereo: device depth is CAM_A-aligned "
                          "(63.7 deg) inside the policy's CAM_B FOV (85.6 deg) "
                          "[유도] and is the x0.64-scaled on-device matcher, "
                          "not the FoundationStereo depth the policy was "
                          "trained on. Pass --fstereo (parity) or "
                          "--policy-allow-device-depth (bench experiment, "
                          "recorded as such).")
    if fs:
        from rov_gui.backends.policy import (fs_settings_mismatch,
                                             training_depth_source)
        ckpt = getattr(a, "policy_ckpt", None)
        tds = None
        if not ckpt:
            try:
                from rov_gui.control.geometry import MpcConfig
                ckpt = MpcConfig.load(getattr(a, "mpc_config",
                                              "config/hw_mpc.yaml")).policy["ckpt"]
            except Exception as e:                               # noqa: BLE001
                warns.append(f"policy: could not read policy.ckpt from the "
                             f"config ({type(e).__name__}: {e}); the FS "
                             f"parity check is skipped")
        if ckpt:
            tds = training_depth_source(ckpt)
            if tds.get("status") == "ok":
                # WARNING, not a refusal, since 2026-09-02: the shipped
                # --policy defaults are themselves off the training store
                # (iters 8 / scale 0.75, chosen for >=10 Hz — see
                # POLICY_DEPENDENT_DEFAULTS), so a refusal here would refuse
                # the default launch. It could not have stayed principled
                # anyway while the same run shipped a 2.07x time-stride
                # mismatch it did not refuse. Both settings and the real pair
                # spacing go into the run meta, which is what keeps the run
                # attributable either way.
                bad = fs_settings_mismatch(a, tds)
    # ---- warnings, all of them
    if not bool(getattr(a, "mpc", False)):
        warns.append("--policy without --mpc: the policy worker loads and "
                     "publishes status, but nothing consumes its plans "
                     "(the `policy` mission shape lives in the --mpc "
                     "controller). Add --mpc to fly it.")
    if src == "hw" and not fs and allow_dev:
        warns.append("policy on DEVICE depth (--policy-allow-device-depth): a "
                     "bench experiment, recorded in meta (depth_scale_applied "
                     "0.64, coverage) — not the training-parity instrument")
    alpha = getattr(a, "fstereo_alpha", None)
    if fs and alpha is not None and float(alpha) < 0.5:
        warns.append(f"policy: --fstereo-alpha {float(alpha):g} < 0.5 — the 224 "
                     f"obs crop is not covered (97.92 % < 98.5 % at alpha 0, "
                     f"the builder REFUSES the grid at run time and the "
                     f"mission would arm without ever receiving a plan); "
                     f"leave alpha at the --policy default 0.5")
    if fs and tds is not None:
        if tds.get("status") == "ok":
            bad = fs_settings_mismatch(a, tds)
            if bad:
                warns.append("policy: flying with FS settings the policy did "
                             "NOT train on — " + "; ".join(bad)
                             + ". This is the shipped speed/parity trade "
                               "(recorded in the meta); pass --fstereo-iters "
                               "16 --fstereo-scale 1.0 for exact instrument "
                               "parity at ~7 Hz. " + FS_RATE_NOTE)
            else:
                warns.append(f"policy: FoundationStereo at the training store's "
                             f"settings (iters {tds['iters']}, scale "
                             f"{tds['scale']:g}, alpha {alpha}). {FS_RATE_NOTE}")
        elif tds.get("status") != "stub":
            warns.append(f"policy: FS-vs-training parity NOT checked "
                         f"({tds.get('status')}{': ' + tds['error'] if tds.get('error') else ''}); "
                         f"flying iters={getattr(a, 'fstereo_iters', None)} "
                         f"scale={getattr(a, 'fstereo_scale', None)}")
    if warns:
        return ("warn", " | ".join(warns))
    return None


def check_fstereo(a) -> str | None:
    """--fstereo only does anything on the hardware source. Say so at launch.

    Only ``HardwareBackend`` builds an FStereoWorker (backends/hardware.py), and
    only the C3 produces a mono pair, so on ``--source demo``/``ros2`` the flag
    yields a depth panel that looks entirely normal and an FS button that does
    nothing at all — while still reading as "I asked for learned depth". That
    is the same class of silent nothing-happens the flag was just fixed for, so
    it is named here rather than discovered in the water.

    A message, or None. This one is a WARNING, not a refusal: rehearsing the
    layout on the demo backend is a legitimate thing to do.
    """
    if not bool(getattr(a, "fstereo", False)):
        return None
    src = getattr(a, "source", "demo")
    if src != "hw":
        return (f"--fstereo does nothing with --source {src}: only the hardware "
                f"backend opens the C3 and builds the learned-depth worker. "
                f"The depth panel will show the {src} source's own depth and "
                f"the FS button is not built.")
    return None


def main(argv: list[str] | None = None) -> int:
    # A native crash (two on 2026-09-02 while three model/solver loads ran at
    # once) says nothing without this; with it the traceback names the thread.
    import faulthandler
    faulthandler.enable()
    a = resolve_defaults(build_parser().parse_args(argv))
    bad = check_fstereo(a)
    if bad:
        print(f"rov_gui: {bad}", file=sys.stderr)
    for check in (check_slam, check_policy):
        got = check(a)
        if not got:
            continue
        level, text = got
        print(f"rov_gui: {'REFUSED — ' if level == 'refuse' else ''}{text}",
              file=sys.stderr)
        if level == "refuse":
            return 2
    if bool(getattr(a, "policy_observe", False)) \
            or str(getattr(a, "mpc_mode", None) or "").lower() == "none":
        # Loud for the land-dry-run reason and one more: this mode DISARMS
        # five automatic stops on a vehicle that really is in the water and
        # really can move. Nothing here can hurt it — nothing is sent — but
        # the operator is now the only thing watching, and they have to know
        # exactly which watchers were switched off on their behalf. Since
        # 2026-09-11 the mode is LOW level = None in the trajectory panel
        # (--policy-observe / --mpc-mode none preselect it at launch); the
        # panel's set_mode logs the same list again at every switch.
        print("\n" + "=" * 78 + "\n"
              "LOW LEVEL = None (teleop; POLICY OBSERVE) — this station will "
              "NOT command the vehicle\n"
              + "=" * 78 + "\n"
              "  * the diffusion policy runs and its plans are DRAWN, never\n"
              "    followed: ctrl.step() is not called and no axis, wrench or\n"
              "    jaw drive leaves this worker. YOU have the joystick, the\n"
              "    whole run — the panel never takes it.\n"
              "  * DISARMED, because each one assumes a follower that is not\n"
              "    there: the divergence guard, the escalation latch, the\n"
              "    bridge-too-long stop, the workspace box, and the tag-loss\n"
              "    disengage. Losing the tag SKIPS plans; it no longer ends\n"
              "    the run and no longer wipes the plot.\n"
              "  * the jaw is forced off regardless of policy.gripper, and\n"
              "    the run clock is policy.observe_max_run_s.\n"
              "  * records go to data/*/*_observe/, never the water\n"
              "    tree; every CSV row carries observe=1, u/w/axes are nan\n"
              "    and solver_status is -1.\n"
              "\n"
              "  NOTHING MEASURED HERE IS A CLOSED-LOOP RESULT. The vehicle\n"
              "  went where you flew it; the reference is what the network\n"
              "  asked for and nobody followed. Do not pool these numbers\n"
              "  with a control run.\n"
              "\n"
              "  You can switch LOW level back to a controller in the panel\n"
              "  between engagements.\n"
              + "=" * 78 + "\n", file=sys.stderr)
    if bool(getattr(a, "land_dry_run", False)):
        # Loud, because this mode relaxes the arm and flight-mode engage gates
        # and the operator has to know why that is safe and what it now depends
        # on THEM for. The one hazard the station cannot close is last.
        print("\n" + "=" * 78 + "\n"
              "LAND DRY-RUN — the vehicle cannot be commanded from this station\n"
              + "=" * 78 + "\n"
              "  * no command sink is built (--allow-command ignored), no MAVLink\n"
              "    rate requests: this process transmits NOTHING to the vehicle\n"
              "  * the localizer is a SYNTHETIC stationary fix — the pose never\n"
              "    moves; a stationary vehicle's proprio is zero either way\n"
              "  * arm / flight-mode / telemetry-stale engage gates are RELAXED,\n"
              "    and so is the divergence guard\n"
              "  * records go to data/*/*_landdry/, never the water tree\n"
              "\n"
              "  DO NOT ARM THE VEHICLE. In any mode but MANUAL the autopilot\n"
              "  spins thrusters by itself, with no byte from here, and nothing\n"
              "  in this station can stop it.\n"
              + "=" * 78 + "\n", file=sys.stderr)
    versions = qt_versions()
    print(f"rov_gui: Qt {versions['qt']} via {versions['api']} {versions['binding']}")

    if a.source == "hw":
        rc = _preflight(a)
        if rc is not None:
            return rc
    elif a.preflight_only:
        print(f"--preflight-only has nothing to check for --source {a.source}: "
              "it opens no hardware.")
        return 0

    from rov_gui.window import build_and_run
    return build_and_run(a)


def _preflight(a) -> int | None:
    """Gate the hardware path. Returns an exit code, or None to continue."""
    try:
        from c3_camera import preflight as PF
    except ImportError as e:
        print(f"could not import c3_camera.preflight ({e}).\n"
              "  --source hw needs depthai 2.x and this repo on sys.path.\n"
              "  Run it through ./c3 gui, which picks the right interpreter,\n"
              "  or pass --no-preflight if you know the rig is ready.",
              file=sys.stderr)
        return None if a.no_preflight else 2
    return PF.gate(a, title="rov_gui", out_dir=Path(a.rec_dir),
                   display=True, vehicle=a.mavlink_transport != "none")


if __name__ == "__main__":
    raise SystemExit(main())
