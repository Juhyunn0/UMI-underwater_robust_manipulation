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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
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
    g.add_argument("--depth-fps", type=float, default=20.0,
                   help="C3 stereo/depth fps, separate from colour because a "
                        "depth frame costs several times a colour one "
                        "(default: %(default)s)")
    g.add_argument("--depth-size", default="640x360", metavar="WxH",
                   help="depth output size (default: %(default)s). Depth is RAW "
                        "uint16 on the wire, so its cost is exactly w*h*2*fps "
                        "with no scene dependence: 640x360@10 measured 36.9 "
                        "Mbit/s against 36.86 predicted. At 20 fps that is "
                        "73.7 Mbit/s — 82%% of the ~90 Mbit/s C3 link on its "
                        "own — so the colour stream has to live in the "
                        "remaining ~16, which is what --isp-scale and "
                        "--mjpeg-quality are for. Shrink this instead if you "
                        "would rather spend the link on the pilot's view.")
    g.add_argument("--no-depth", dest="depth", action="store_false",
                   help="do not compute or stream depth at all — the cheapest "
                        "way to give the colour feed the whole link")
    g.add_argument("--panel2", default="rov", choices=("rov", "stereo", "none"),
                   help="what the middle panel shows: the ROV's own RGB camera "
                        "over BlueOS RTP, or the C3's left mono image "
                        "(default: %(default)s)")
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
                        "(sessions/ui_recordings/c3_depth_20260809_202409_"
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
    g.add_argument("--pose-ref-dir", default="sessions/pose_meshes",
                   metavar="DIR",
                   help="parent directory for on-site reconstructions "
                        "(default: %(default)s). Each attempt gets its own "
                        "obj_<timestamp>/ holding the reference views, the "
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
                   choices=("mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned",
                            "pid"),
                   help="override hw_mpc.yaml mode. One solver serves all "
                        "four MPC names: the 'dob' prefix adds the EAOB "
                        "disturbance feedforward, the '_tuned' suffix splits "
                        "the position cost into along-track (cheap) and "
                        "cross-track (expensive) in the path frame "
                        "(hw_mpc.yaml mpc_tuned:)")
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

    g = p.add_argument_group("ui")
    g.add_argument("--ui-fps", type=float, default=60.0,
                   help="repaint rate (default: %(default)s). This is real "
                        "display latency: at 30 Hz a frame waits up to 33 ms "
                        "for the next paint, on top of the camera's own 34 ms.")
    g.add_argument("--fullscreen", action="store_true")
    g.add_argument("--rec-dir", default="sessions/low_level_controller_data",
                   help="ROOT of the dated run tree (default: %(default)s). "
                        "Everything one run produces — the UI recording, the "
                        "feed recordings, the nav recording, the controller "
                        "CSV and events.log — lands together in "
                        "<rec-dir>/YYYYMMDD/MMDD_HHMMSS/. See "
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


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
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
