#!/usr/bin/env python3
"""
Keyboard teleop + live force-arrow visualization for the BlueROV2 (FLU).

Keys set a latched FLU body wrench -> pinv(B) -> 6 thruster forces -> data.ctrl
(only the 5 controllable DOFs; pitch is underactuated and never commanded):

    W / S : surge  +x / -x      A / D : yaw  +Mz / -Mz
    Q / E : sway   +y / -y      Z / C : roll +Mx / -Mx
    R / F : heave  +z / -z      X     : STOP (zero thrust)
    G     : toggle disturbances (current + waves + kicks)
    Ctrl-C / close window : quit

Two viewer modes:
  * DEFAULT (launch_passive): draws live color-coded FORCE ARROWS on the vehicle
    (buoyancy/drag/thrust/kick + current/wave flow). Keys come from the FOCUSED
    VIEWER WINDOW -- click the viewer, then drive.
        - Ubuntu (target): `python teleop.py`            (plain python)
        - macOS (preview): `mjpython teleop.py`          (from no-space venv
          ~/bluerov_venv; the project path's space breaks mjpython otherwise)
  * --managed: the old managed viewer (no arrows), runs with plain `python`
    anywhere; keys come from the focused TERMINAL. Handy for quick macOS checks
    without mjpython.

The arrow viz only READS the forces already computed by hydro.py/disturbances.py/
thrusters.py -- it does not change the dynamics.

    python teleop.py --disturb   # start with disturbances on (toggle with G)
    python teleop.py --plot      # add console sparklines of drag/wave/kick
    python teleop.py --scale 1.5 # scale command magnitudes
    python teleop.py --selftest  # headless: assert each key -> correct FLU direction

Gravity + hydro (Phase 3) are ON by default; --no-hydro reverts to thruster-only,
gravity-off. Only `mujoco` + `numpy` required (`pynput` optional, --managed only).
No mjx/jax/cuda.
"""
import argparse
import collections
import os
import sys
import threading
import time

import numpy as np
import mujoco
import mujoco.viewer   # import is display-safe; only launch*() needs a display

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thrusters as T
import hydro as H
import disturbances as D

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "bluerov.xml")

# Fixed command magnitudes per DOF (scaled by --scale). Surge is kept gentle on
# purpose: with hydro ON, a strong surge force (applied 0.0725 m below the COM)
# makes the vehicle nose-down/over because the weak coBM=0.01 restoring can't
# counter it (uncontrollable pitch). Sway/heave/yaw/roll have no such limit.
SURGE_N = 8.0       # Fx
SWAY_N = 15.0       # Fy
HEAVE_N = 20.0      # Fz
YAW_NM = 6.0        # Mz
ROLL_NM = 3.0       # Mx

# ---- force-arrow visualization constants ----
# Arrow length = scale * magnitude, capped, so big forces don't make giant arrows.
FORCE_SCALE = 0.003   # m per N   (buoyancy 111 N -> 0.33 m, ~ vehicle size)
FORCE_CAP = 0.6       # m         (cap so e.g. a 180 N net thrust stays readable)
VEL_SCALE = 0.5       # m per m/s (current 0.2 m/s -> 0.10 m)
VEL_CAP = 0.4         # m
ARROW_W = 0.012       # m         (shaft half-width)
COLORS = {            # color code (rgba); documented in the legend
    "buoyancy": (0.10, 0.80, 0.20, 1.0),   # green
    "drag":     (0.65, 0.65, 0.65, 1.0),   # gray
    "thrust":   (1.00, 0.55, 0.00, 1.0),   # orange
    "kick":     (0.95, 0.10, 0.10, 1.0),   # red
    "current":  (0.20, 0.45, 1.00, 1.0),   # blue
    "wave":     (0.10, 0.85, 0.90, 1.0),   # cyan
}
LEGEND = "buoyancy=green  drag=gray  thrust=orange  kick=red  current=blue  wave=cyan"


# wrench index order: [Fx, Fy, Fz, Mx, My, Mz]   (My = pitch is never set)
def keymap(scale=1.0):
    return {
        "W": (0, +SURGE_N * scale), "S": (0, -SURGE_N * scale),
        "Q": (1, +SWAY_N * scale),  "E": (1, -SWAY_N * scale),
        "R": (2, +HEAVE_N * scale), "F": (2, -HEAVE_N * scale),
        "A": (5, +YAW_NM * scale),  "D": (5, -YAW_NM * scale),
        "Z": (3, +ROLL_NM * scale), "C": (3, -ROLL_NM * scale),
    }


HELP = """\
=================== BlueROV2 keyboard teleop (FLU) ===================
  W / S : surge  +x / -x   (forward / back)
  Q / E : sway   +y / -y   (left / right ; FLU +y = PORT)
  R / F : heave  +z / -z   (up / down)
  A / D : yaw   +Mz / -Mz  (turn left / right)
  Z / C : roll  +Mx / -Mx
  X     : STOP  (zero thrust; drag brings you to rest)
  G     : toggle disturbances (current + waves + kicks)
  Ctrl-C / close window : quit
  Commands latch until changed.  Gravity + hydro ON (--no-hydro to disable).
=====================================================================\
"""


class Teleop:
    """Shared control state: keys -> latched FLU wrench -> thruster forces."""

    def __init__(self, model, data, scale=1.0, verbose=True, disturbance=None):
        self.model, self.data = model, data
        self.B, _ = T.allocation_matrix(model, data)
        self.pinvB = np.linalg.pinv(self.B)
        self.keymap = keymap(scale)
        self.wrench = np.zeros(6)
        self.lock = threading.Lock()
        self.verbose = verbose
        self.disturbance = disturbance       # Phase 4: G toggles current/waves/kicks

    def on_key(self, ch):
        ch = (ch or "").upper()
        if ch == "G" and self.disturbance is not None:
            on = self.disturbance.toggle()
            sys.stdout.write(f"\n[disturbances {'ON' if on else 'OFF'}: "
                             f"current+waves+kicks]\n")
            sys.stdout.flush()
            return
        with self.lock:
            if ch == "X":
                self.wrench[:] = 0.0
            elif ch in self.keymap:
                idx, val = self.keymap[ch]
                self.wrench[idx] = val
            else:
                return
            forces = self.pinvB @ self.wrench
        T.set_thruster_forces(self.model, self.data, forces)
        if self.verbose:
            w = self.wrench
            applied = np.array(self.data.ctrl)
            sys.stdout.write(
                f"\r[surge {w[0]:+5.0f} sway {w[1]:+5.0f} heave {w[2]:+5.0f} "
                f"yaw {w[5]:+4.0f} roll {w[3]:+4.0f}]  thrust(N)="
                f"{np.array2string(applied, precision=1, sign='+')}      ")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# force arrows (launch_passive only)
# ---------------------------------------------------------------------------
def _add_arrow(scn, frm, vec, scale, color, cap, label="", min_mag=0.0):
    mag = float(np.linalg.norm(vec))
    if mag <= min_mag or scn.ngeom >= scn.maxgeom:
        return
    frm = np.asarray(frm, dtype=float)
    to = frm + (np.asarray(vec, dtype=float) / mag) * min(scale * mag, cap)
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_W, frm, to)
    if label:
        g.label = label
    scn.ngeom += 1


def draw_force_arrows(scn, hydro, teleop, data, bid):
    """Populate user_scn with one arrow per force component. Returns magnitudes."""
    scn.ngeom = 0
    mags = {}
    R = data.xmat[bid].reshape(3, 3)
    com = data.xipos[bid].copy()

    # net thrust (orange), computed from the live ctrl + allocation B
    f_thr = R @ (teleop.B[:3] @ np.array(data.ctrl))
    mags["thrust"] = float(np.linalg.norm(f_thr))
    _add_arrow(scn, com, f_thr, FORCE_SCALE, COLORS["thrust"], FORCE_CAP,
               f"thrust {mags['thrust']:.0f}N", min_mag=0.5)

    if hydro is not None and hydro.components:
        for name in ("buoyancy", "drag", "kick"):          # forces (N)
            pt, vec = hydro.components[name]
            mags[name] = float(np.linalg.norm(vec))
            _add_arrow(scn, pt, vec, FORCE_SCALE, COLORS[name], FORCE_CAP,
                       f"{name} {mags[name]:.0f}N", min_mag=0.5)
        for name in ("current", "wave"):                   # water velocity (m/s)
            pt, vel = hydro.water[name]
            mags[name] = float(np.linalg.norm(vel))
            _add_arrow(scn, pt, vel, VEL_SCALE, COLORS[name], VEL_CAP,
                       f"{name} {mags[name]:.2f}m/s", min_mag=0.01)
    return mags


_SPARK = "▁▂▃▄▅▆▇█"


def _spark(vals):
    if not vals:
        return ""
    hi = max(max(vals), 1e-9)
    return "".join(_SPARK[min(7, int(v / hi * 7.999))] for v in vals)


def _status(teleop, mags, hist):
    w = teleop.wrench
    line = (f"\rcmd[su{w[0]:+.0f} sw{w[1]:+.0f} hv{w[2]:+.0f} yw{w[5]:+.0f} "
            f"rl{w[3]:+.0f}]  F(N): thr{mags.get('thrust',0):4.0f} "
            f"buoy{mags.get('buoyancy',0):4.0f} drag{mags.get('drag',0):4.0f} "
            f"kick{mags.get('kick',0):3.0f}  flow(m/s): cur{mags.get('current',0):.2f} "
            f"wav{mags.get('wave',0):.2f}")
    if hist is not None:
        line += (f"  |drag {_spark(hist['drag'])} wav {_spark(hist['wave'])} "
                 f"kick {_spark(hist['kick'])}")
    sys.stdout.write(line + "   ")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# viewer modes
# ---------------------------------------------------------------------------
def run_passive(model, data, teleop, hydro, args):
    """launch_passive: own step+sync loop, live force arrows, key_callback."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    dt = model.opt.timestep
    n_sub = max(1, round((1.0 / 60.0) / dt))      # ~60 fps render
    hist = {k: collections.deque(maxlen=48) for k in ("drag", "wave", "kick")}

    def key_cb(keycode):
        if 32 <= keycode < 127:                   # printable -> letter keys
            teleop.on_key(chr(keycode))

    print("Force arrows ON.  Legend:  " + LEGEND)
    print(f"  length = magnitude (force {FORCE_SCALE} m/N cap {FORCE_CAP} m; "
          f"velocity {VEL_SCALE} m per m/s cap {VEL_CAP} m).")
    print("Keys come from the FOCUSED VIEWER WINDOW — click the viewer, then drive.")
    last = [0.0]
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            t0 = time.time()
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            mags = draw_force_arrows(viewer.user_scn, hydro, teleop, data, bid)
            viewer.sync()
            if args.plot:
                for k in hist:
                    hist[k].append(mags.get(k, 0.0))
            if t0 - last[0] > 0.12:
                _status(teleop, mags, hist if args.plot else None)
                last[0] = t0
            slack = n_sub * dt - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)
    print("\nbye")


def run_managed(model, data, teleop, args):
    """Old managed viewer (no arrows); keys from the focused TERMINAL."""
    backend = run_pynput if args.pynput else run_stdin
    if args.pynput:
        print("Key capture: pynput (global). If keys do nothing on macOS, grant "
              "Accessibility to your terminal or omit --pynput.")
    else:
        print("Key capture: terminal — keep THIS terminal window focused (not the "
              "viewer) while driving; click the viewer only to move the camera.")
    stop = threading.Event()
    kbd = threading.Thread(target=backend, args=(teleop, stop), daemon=True)
    kbd.start()
    try:
        mujoco.viewer.launch(model, data)         # GUI on main thread (plain python)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    print("\nbye")


# ---------------------------------------------------------------------------
# keyboard backends for --managed (run in a background thread)
# ---------------------------------------------------------------------------
def run_stdin(teleop, stop):
    """Read keys from the terminal in cbreak mode (no extra deps).

    Keep THIS terminal focused while driving. cbreak leaves ISIG on, so Ctrl-C
    still raises KeyboardInterrupt in the main thread to quit cleanly.
    """
    import termios
    import tty
    import select
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch:
                    teleop.on_key(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_pynput(teleop, stop):
    """Global key capture via pynput (works even when the viewer is focused)."""
    from pynput import keyboard

    def on_press(key):
        ch = getattr(key, "char", None)
        if ch:
            teleop.on_key(ch)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    stop.wait()
    listener.stop()


# ---------------------------------------------------------------------------
def selftest(model, data, scale=1.0):
    """Headless: press each key, assert the resulting accel is the right FLU DOF."""
    model.opt.gravity[:] = 0.0
    tp = Teleop(model, data, scale=scale, verbose=False)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    m = model.body_mass[bid]
    I = np.array(model.body_inertia[bid])

    def accel6(ch):
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        tp.wrench[:] = 0.0
        tp.on_key(ch)
        mujoco.mj_forward(model, data)
        return np.concatenate([m * np.array(data.qacc[:3]),
                               I * np.array(data.qacc[3:6])])  # [F; M]

    expect = {"W": (0, +1), "S": (0, -1), "Q": (1, +1), "E": (1, -1),
              "R": (2, +1), "F": (2, -1), "A": (5, +1), "D": (5, -1),
              "Z": (3, +1), "C": (3, -1)}
    names = {0: "surge +x", 1: "sway +y", 2: "heave +z",
             3: "roll Mx", 4: "pitch My", 5: "yaw Mz"}
    print(HELP)
    print("\nSelf-test (gravity off): each key -> dominant FLU response\n")
    ok_all = True
    for ch, (dom, sgn) in expect.items():
        w = accel6(ch)
        dom_meas = int(np.argmax(np.abs(w)))
        ok = (dom_meas == dom) and (np.sign(w[dom]) == sgn) and (dom != 4)
        ok_all &= ok
        print(f"  {ch}: wrench={np.array2string(w, precision=2, sign='+'):44s}"
              f" -> {names[dom_meas]:9s} {'PASS' if ok else 'FAIL'}")

    mujoco.mj_resetData(model, data)
    tp.on_key("W")
    tp.on_key("X")
    x_ok = np.allclose(data.ctrl, 0.0)
    ok_all &= x_ok
    print(f"  X: all thrust -> {np.array2string(np.array(data.ctrl), precision=1)}"
          f"  {'PASS' if x_ok else 'FAIL'}")
    print("\n" + ("TELEOP SELF-TEST PASSED" if ok_all else "TELEOP SELF-TEST FAILED"))
    assert ok_all, "teleop key->direction mapping failed"


def main():
    ap = argparse.ArgumentParser(description="BlueROV2 keyboard teleop (FLU) + force arrows.")
    ap.add_argument("--managed", action="store_true",
                    help="old managed viewer (no arrows); plain python anywhere, terminal keys")
    ap.add_argument("--plot", action="store_true",
                    help="console sparklines of drag/wave/kick magnitude over time")
    ap.add_argument("--pynput", action="store_true",
                    help="(--managed only) global key capture via pynput")
    ap.add_argument("--selftest", action="store_true",
                    help="headless check of key->direction mapping, then exit")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale all command magnitudes (default 1.0)")
    ap.add_argument("--no-hydro", action="store_true",
                    help="disable Phase-3 hydro + gravity (thruster-only mode)")
    ap.add_argument("--disturb", action="store_true",
                    help="start with Phase-4 disturbances ON (toggle with G)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)

    if args.selftest:
        selftest(model, data, scale=args.scale)
        return

    field = None
    if args.no_hydro:
        model.opt.gravity[:] = 0.0          # thruster-only mode (Phase 1/2 feel)
        hydro = None
    else:
        field = D.DisturbanceField(seed=0)  # Phase 4: current+waves+kicks (toggle G)
        field.enabled = args.disturb
        hydro = H.Hydrodynamics(model, disturbance=field).install()  # gravity ON
    # --managed prints its status from on_key (terminal); passive prints from loop
    teleop = Teleop(model, data, scale=args.scale, verbose=args.managed,
                    disturbance=field)

    print(HELP)
    print(("Hydro+gravity ON (in-water feel: drag stops you, buoyancy holds depth, "
           "tilt self-rights)." if hydro else
           "Hydro OFF, gravity OFF (thruster-only mode)."))
    if field is not None:
        print(f"Disturbances {'ON' if field.enabled else 'OFF'} (press G to toggle). "
              f"current |{np.linalg.norm(field.current):.2f}| m/s, waves, kicks; "
              f"depth ~{field.z_surface:.0f} m at start.")

    if args.managed:
        run_managed(model, data, teleop, args)
    else:
        run_passive(model, data, teleop, hydro, args)


if __name__ == "__main__":
    main()
