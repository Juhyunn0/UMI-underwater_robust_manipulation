#!/usr/bin/env python3
"""
Keyboard teleop + live force-arrow visualization for the BlueROV2 (FLU).

Keys set a latched FLU body wrench -> pinv(B) -> 6 thruster forces -> data.ctrl
(only the 5 controllable DOFs; pitch is underactuated and never commanded):

    W / S : surge  +x / -x      A / D : yaw  +Mz / -Mz
    Q / E : sway   +y / -y      Z / C : roll +Mx / -Mx
    R / F : heave  +z / -z      X     : STOP (zero thrust)
    G     : toggle disturbances (current + waves + kicks)
    V     : toggle force/flow arrows (hide the overlay; --no-arrows starts hidden)
    Ctrl-C / close window : quit

Three viewer modes:
  * DEFAULT (launch_passive): draws live color-coded FORCE ARROWS on the vehicle
    (buoyancy/drag/thrust/kick + current/wave flow). Keys come from the FOCUSED
    VIEWER WINDOW -- click the viewer, then drive.
        - Ubuntu (target): `python teleop.py`            (plain python)
        - macOS (preview): `mjpython teleop.py`          (from no-space venv
          ~/bluerov_venv; the project path's space breaks mjpython otherwise)
  * --managed: the old managed viewer (no arrows), runs with plain `python`
    anywhere; keys come from the focused TERMINAL. Handy for quick macOS checks
    without mjpython.
  * --viser / --remote: HEADLESS web UI -- NO GLFW. Starts a viser server on
    :8080 (host 0.0.0.0, so reachable over SSH/Tailscale from the MacBook),
    rebuilds the model from mjModel+mjData each frame, and exposes the W/A/S/D...
    controls as browser GUI buttons/sliders (plus the same force arrows). Use
    this when there is no local display.
        - `python teleop.py --viser`     then open http://<host>:8080

The arrow viz only READS the forces already computed by hydro.py/disturbances.py/
thrusters.py -- it does not change the dynamics.

    python teleop.py --disturb   # start with disturbances on (toggle with G)
    python teleop.py --plot      # add console sparklines of drag/wave/kick
    python teleop.py --scale 1.5 # scale command magnitudes
    python teleop.py --viser     # headless: drive from a browser (SSH/Tailscale)
    python teleop.py --selftest  # headless: assert each key -> correct FLU direction
    python teleop.py --observe   # DON'T pilot: release the ROV from rest and watch the
                                 # current+waves carry it (disturbances forced ON; drive
                                 # keys disabled; H recenters; auto-recenter when it drifts
                                 # out of view). Best with POOL_TAGS=1 for the water surface.

Gravity + hydro (Phase 3) are ON by default; --no-hydro reverts to thruster-only,
gravity-off. Only `mujoco` + `numpy` required for the local modes (`pynput`
optional, --managed only; `viser` + `trimesh` only for --viser). No mjx/jax/cuda.
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
# The interactive teleop VIEWER defaults to the pool AprilTag floor + animated water
# surface (POOL_TAGS), since that's the context you actually watch. It's VISUAL ONLY
# (dynamics verified Δ=0), so this changes nothing physical. setdefault means an explicit
# `POOL_TAGS=0` still wins (opt out for a bare scene), and the headless experiment/test/
# verify entry points (run_compare, eval_dp, verify_*, test_*) are UNAFFECTED — they read
# the env directly (default off) and keep their clean plant baseline. Must run before the
# rov_model import, which fixes XML_PATH from POOL_TAGS at import time.
os.environ.setdefault("POOL_TAGS", "1")
import thrusters as T
import hydro as H
import disturbances as D
import rov_model as RM     # which BlueROV variant (env ROV_MODEL): bluerov2 | heavy
import water_viz as WV     # animated pool water surface (VISUAL ONLY; POOL_TAGS scene)

HERE = os.path.dirname(os.path.abspath(__file__))
XML = RM.XML_PATH          # bluerov.xml (6 thr) or bluerov_heavy.xml (8 thr)

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
# Arrow length = scale * magnitude, capped (big forces don't make giant arrows) and
# floored (small forces stay visible). Thick shaft + a tip marker carrying the label.
FORCE_SCALE = 0.006   # m per N   (2x: mid forces like thrust/kick now clearly sized)
FORCE_CAP = 0.6       # m         (cap so e.g. buoyancy 111 N stays ~ vehicle size)
VEL_SCALE = 0.8       # m per m/s (current 0.2 m/s -> 0.16 m)
VEL_CAP = 0.4         # m
ARROW_W = 0.02        # m         (shaft half-width; thicker = easier to see)
MIN_LEN = 0.06        # m         (length floor so small-but-present forces show)
LABEL_DOT_R = 0.012   # m         (tiny tip marker that carries the arrow's label)
COLORS = {            # color code (rgba); documented in the legend
    "buoyancy": (0.10, 0.80, 0.20, 1.0),   # green
    "drag":     (0.65, 0.65, 0.65, 1.0),   # gray
    "thrust":   (1.00, 0.55, 0.00, 1.0),   # orange
    "kick":     (0.95, 0.10, 0.10, 1.0),   # red
    "current":  (0.20, 0.45, 1.00, 1.0),   # blue
    "wave":     (0.10, 0.85, 0.90, 1.0),   # cyan
}
SHORT = {             # compact label names (the color legend carries the full name)
    "buoyancy": "buoy", "drag": "drag", "thrust": "thr",
    "kick": "kick", "current": "cur", "wave": "wav",
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
  V     : toggle force/flow arrows (hide the overlay)
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
        self.show_arrows = True              # V toggles the force/flow arrow overlay
        # --observe (free-drift) mode: the ROV is UNCONTROLLED — drive keys are ignored
        # and the viewer loop watches the disturbance carry it. H requests a recenter,
        # which the loop (the physics thread) performs so it never races MjData.
        self.observe = False
        self._recenter_request = False
        # random wave/current headings (set up in main for --observe): N re-draws
        self._dir_random = False
        self._redraw_request = False
        self._beta = 0.0            # current wave mean heading [deg]
        self._theta = 0.0           # current heading theta_c [deg]
        self._dir_draw = None       # () -> (beta_deg, theta_deg)
        self._dir_apply = None      # (beta_deg, theta_deg) -> None (mutates the field)

    def on_key(self, ch):
        ch = (ch or "").upper()
        if ch == "G" and self.disturbance is not None:
            on = self.disturbance.toggle()
            sys.stdout.write(f"\n[disturbances {'ON' if on else 'OFF'}: "
                             f"current+waves+kicks]\n")
            sys.stdout.flush()
            return
        if ch == "V":                        # toggle force/flow arrows (works in every mode)
            self.show_arrows = not self.show_arrows
            sys.stdout.write(f"\n[force arrows {'ON' if self.show_arrows else 'OFF'}]\n")
            sys.stdout.flush()
            return
        if self.observe:                     # uncontrolled: no piloting, only recenter/redraw
            if ch == "H":
                self._recenter_request = True
                sys.stdout.write("\n[observe] recenter -> release pose\n")
                sys.stdout.flush()
            elif ch == "N" and getattr(self, "_dir_random", False):
                self._redraw_request = True
                sys.stdout.write("\n[observe] re-draw random wave/current headings\n")
                sys.stdout.flush()
            return                           # ignore W/S/Q/E/R/F/A/D/Z/C/X
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
    length = max(min(scale * mag, cap), MIN_LEN)   # cap big, floor small -> all visible
    to = frm + (np.asarray(vec, dtype=float) / mag) * length
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_W, frm, to)
    scn.ngeom += 1
    # Attach the label to a tiny marker at the arrow TIP, not to the arrow geom (whose
    # text renders back at the shared COM start point, so every label piles up there).
    # The tips fan out with the arrows, so the labels separate instead of overlapping.
    if label and scn.ngeom < scn.maxgeom:
        gd = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(gd, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, LABEL_DOT_R),
                            to, np.eye(3).flatten(), np.asarray(color, dtype=np.float32))
        gd.label = label
        scn.ngeom += 1


def _draw_plan(scn, pts, color=(1.0, 0.82, 0.3, 1.0), width=0.004):
    """Append a planned path (polyline `pts`, (N,3)) to user_scn as capsule segments."""
    for i in range(len(pts) - 1):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.asarray(color, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.asarray(pts[i], float), np.asarray(pts[i + 1], float))
        scn.ngeom += 1


def _controller_meta(controller, ctrl_name=None):
    """Compact controller descriptor for the per-run manifest. Reports the ACTUAL
    NMPC solver in use (acados vs ipopt -- detected from the live solver object, so
    it reflects a factory fallback), the horizon N, the control rate, the mode, and
    for a PoseController the effective merged gain set (DEFAULT_GAINS + overrides)."""
    if controller is None:
        return {"type": "teleop"}
    m = {"type": ctrl_name or getattr(controller, "mode", "?")}
    mode = getattr(controller, "mode", None)
    if mode:
        m["mode"] = mode                                    # dobmpc (w_hat on) vs mpc
    gains = getattr(controller, "g", None)
    if gains:                                               # PoseController: merged set
        m["gains"] = dict(gains)
    nmpc = getattr(controller, "nmpc", None)
    if nmpc is not None:
        m["solver"] = "acados" if type(nmpc).__name__ == "AcadosNMPC" else "ipopt"
        m["N"] = int(getattr(nmpc, "N", 0))
    cd = getattr(controller, "ctrl_dt", None)
    if cd:
        m["ctrl_hz"] = round(1.0 / cd, 3)
    return m


def _run_manifest(field, controller, args, model, trajectory):
    """Full per-run manifest (disturbance + controller/solver + trajectory + run
    context) for the CSV sidecar. Built lazily at recording start so field.to_meta()
    snapshots the live enabled/seed/kick state."""
    from recorder import build_run_meta
    if getattr(args, "ideal_thrusters", False):
        thrusters = dict(model="ideal")                     # commanded == realized
    else:
        thrusters = dict(model="T200_realistic", lag=True,
                         voltage_scale=float(getattr(args, "thruster_voltage",
                                                     T.NOMINAL_VOLTAGE_SCALE)))
    return build_run_meta(
        disturbance=field,
        controller=_controller_meta(controller, getattr(args, "ctrl", None)),
        trajectory=trajectory,
        run=dict(started=time.strftime("%Y-%m-%d %H:%M:%S"),
                 sim_dt=float(model.opt.timestep),
                 waves=getattr(args, "waves", None),
                 thrusters=thrusters),
    )


def _plan_points(mission=None, controller=None):
    """Planned-trajectory points (N,3) for the monitor's 2D projections, or None.
    A mission supplies its path (e.g. the square); a goto-origin controller supplies
    its setpoint as a single target marker; plain teleop has no plan."""
    if mission is not None and hasattr(mission, "plan_points"):
        return np.asarray(mission.plan_points(), float)
    if controller is not None and hasattr(controller, "p_ref"):
        return np.asarray(controller.p_ref, float).reshape(1, 3)
    return None


def force_items(hydro, teleop, data, bid):
    """Every force/flow arrow as (name, point, vector, scale, color, cap, min_mag)
    in world coordinates. Single source of truth shared by the local (mjv) drawer,
    the viser drawer, and the status line. READ-ONLY: it reflects forces already
    computed by the physics; it does not change the dynamics.

    Forces use min_mag>=0.5 N (labelled in N); flows use min_mag<0.5 (m/s).
    """
    R = data.xmat[bid].reshape(3, 3)
    com = data.xipos[bid].copy()
    # net thrust (orange), computed from the live ctrl + allocation B
    items = [("thrust", com, R @ (teleop.B[:3] @ np.array(data.ctrl)),
              FORCE_SCALE, COLORS["thrust"], FORCE_CAP, 0.5)]
    if hydro is not None and hydro.components:
        for name in ("buoyancy", "drag", "kick"):          # forces (N)
            pt, vec = hydro.components[name]
            items.append((name, np.asarray(pt, float), np.asarray(vec, float),
                          FORCE_SCALE, COLORS[name], FORCE_CAP, 0.5))
        for name in ("current", "wave"):                   # water velocity (m/s)
            pt, vel = hydro.water[name]
            items.append((name, np.asarray(pt, float), np.asarray(vel, float),
                          VEL_SCALE, COLORS[name], VEL_CAP, 0.01))
    return items


def draw_force_arrows(scn, hydro, teleop, data, bid):
    """Populate user_scn with one arrow per force component. Returns magnitudes."""
    scn.ngeom = 0
    mags = {}
    for name, pt, vec, scale, color, cap, min_mag in force_items(hydro, teleop, data, bid):
        mags[name] = float(np.linalg.norm(vec))
        sn = SHORT.get(name, name)
        label = (f"{sn} {mags[name]:.0f}N" if min_mag >= 0.5
                 else f"{sn} {mags[name]:.2f}m/s")
        _add_arrow(scn, pt, vec, scale, color, cap, label, min_mag=min_mag)
    return mags


def force_magnitudes(hydro, teleop, data, bid):
    """Just {name: |vector|} (for the viser status panel; no scene needed)."""
    return {name: float(np.linalg.norm(vec))
            for name, _pt, vec, *_ in force_items(hydro, teleop, data, bid)}


def monitor_sample(hydro, data, bid):
    """Read-only per-frame sample for the --monitor dashboard: total disturbance
    water velocity (current + wave, m/s, world FLU) + ROV COM world position (m).
    Zeros when hydro/disturbance is off. Plain floats so it pickles cheaply to the
    dashboard process."""
    pos = tuple(float(x) for x in data.xipos[bid])
    if hydro is not None and hydro.water:
        vtot = (np.asarray(hydro.water["current"][1], float)
                + np.asarray(hydro.water["wave"][1], float))
    else:
        vtot = np.zeros(3)
    return {"t": float(data.time), "vtot": tuple(map(float, vtot)), "pos": pos}


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
# observe (free-drift) mode helpers
# ---------------------------------------------------------------------------
def _current_from_heading(speed, theta_deg):
    """FLU current vector (m/s) at horizontal heading theta_c [deg]."""
    th = np.radians(theta_deg)
    return float(speed) * np.array([np.cos(th), np.sin(th), 0.0])


def _waves_at_heading(args, beta_deg):
    """Legacy wave-component list with mean heading beta [deg]. Spectrum: JONSWAP about
    beta (same sea/seed, just rotated). Classic: the 3 default sinusoids rigidly rotated."""
    if args.waves == "spectrum":
        hs, tp = (float(v) for v in args.sea.split(","))
        return D.jonswap_wave_specs(Hs=hs, Tp=tp, heading_deg=beta_deg, seed=0)
    return [dict(w, heading_deg=w["heading_deg"] + beta_deg) for w in D.DEFAULT_WAVES]


def _apply_directions(field, args, beta_deg, theta_deg, speed):
    """Set the disturbance field's wave heading (beta) + current heading/speed (theta_c)
    IN PLACE — a scenario setting, not a plant change. hydro reads field.current/.waves
    live each step, so the new headings take effect next step. Rebuilds the wave tuples by
    borrowing a throwaway field's (avoids touching disturbances.py)."""
    field.waves = D.DisturbanceField(waves=_waves_at_heading(args, beta_deg), seed=0).waves
    field.current = _current_from_heading(speed, theta_deg)


def _water_bounds(model, margin=0.35):
    """AABB of the VISUAL water volume (the `pool_water_surface` geom), or None if the
    scene has no water (bare POOL_TAGS=0). Used as the default --observe recenter
    boundary: the ROV may drift anywhere INSIDE the water; leaving it triggers the
    recenter. Horizontal half-extents are inset by `margin` so the vehicle recenters
    while still fully inside the water; the vertical span runs from the seabed skirt
    to the calm waterline (the sim has no free-surface physics, so a positively
    buoyant ROV would otherwise rise straight out of the water visual). Read-only.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pool_water_surface")
    if gid < 0:
        return None
    cx, cy, pz = (float(v) for v in model.geom_pos[gid])
    if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_HFIELD):
        hfid = int(model.geom_dataid[gid])
        rx, ry, elev, base = (float(v) for v in model.hfield_size[hfid])
        z_top = pz + 0.5 * elev            # calm waterline (hfield d=0.5)
        z_bot = pz - base
    else:                                  # flat-box water (--no-water-anim)
        rx, ry, hz = (float(v) for v in model.geom_size[gid][:3])
        z_top, z_bot = pz + hz, pz - hz
    return (cx, cy, max(0.1, rx - margin), max(0.1, ry - margin), z_bot, z_top)


def _maybe_recenter(model, data, hydro, home, radius, force=False, bounds=None):
    """Snap the freely-drifting ROV back to its release pose so it stays in view.

    This is a VIEWING convenience, NOT a dynamics change: it only writes state
    (qpos/qvel) — exactly like the viewer's "Reset pose" button / mj_resetData —
    and never touches the physics model, hydro coefficients, or the disturbance
    field. Trigger: with `bounds` (the water-volume AABB from _water_bounds, the
    default) the ROV recenters when it LEAVES the water; otherwise when the
    horizontal drift OR depth change from the release pose exceeds `radius`.
    `force` (a manual H press) recenters immediately. Returns True if it
    recentered. Called from the viewer loop (the physics thread) so it can't race
    MjData. The disturbance clock keeps running, so each release sees a fresh wave
    phase.
    """
    home_qpos, _ = home
    p = np.asarray(data.qpos[:3], float)
    if bounds is not None:
        cx, cy, rx, ry, z_bot, z_top = bounds
        out = (abs(p[0] - cx) > rx or abs(p[1] - cy) > ry
               or p[2] > z_top or p[2] < z_bot)
    else:
        dp = p - np.asarray(home_qpos[:3])
        out = float(np.hypot(dp[0], dp[1])) > radius or abs(float(dp[2])) > radius
    if force or out:
        data.qpos[:] = home_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)      # refresh xpos/xmat for this frame's arrows
        if hydro is not None:
            hydro.reset()                   # clear the added-mass filter lag (state only)
        return True
    return False


def _status_observe(teleop, data, home, mags):
    """One-line status for --observe: wave/current headings + drift from the release pose +
    the water flow and hydro forces acting on the (uncommanded) vehicle. No cmd wrench."""
    dp = np.asarray(data.qpos[:3]) - np.asarray(home[0][:3])
    horiz = float(np.hypot(dp[0], dp[1]))
    beta = getattr(teleop, "_beta", 0.0)
    theta = getattr(teleop, "_theta", 0.0)
    sys.stdout.write(
        f"\r[observe] wavβ{beta:3.0f}° curθ{theta:3.0f}°  drift H{horiz:4.2f}m "
        f"z{dp[2]:+5.2f}m  flow(m/s): cur{mags.get('current', 0):.2f} wav{mags.get('wave', 0):.2f}"
        f"  F(N): buoy{mags.get('buoyancy', 0):4.0f} drag{mags.get('drag', 0):4.0f} "
        f"kick{mags.get('kick', 0):3.0f}   ")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# viewer modes
# ---------------------------------------------------------------------------
def run_passive(model, data, teleop, hydro, args, monitor=None, controller=None,
                mission=None):
    """launch_passive: own step+sync loop, live force arrows, key_callback."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    dt = model.opt.timestep
    n_sub = max(1, round((1.0 / 60.0) / dt))      # ~60 fps render
    hist = {k: collections.deque(maxlen=48) for k in ("drag", "wave", "kick")}
    last_mon = [0.0]
    surf = WV.make_surface_from_env(model)        # animated water (None unless POOL_TAGS hfield scene)
    if surf is not None:
        print("Animated water surface ON (VISUAL ONLY; waves+current from the disturbance field; "
              "flat when disturbances are off).")
    observe = getattr(teleop, "observe", False)   # free-drift: uncontrolled, watch the flow
    home = (data.qpos.copy(), data.qvel.copy())    # release pose for recenter
    # Recenter boundary: default = the WATER volume itself (drift anywhere inside the
    # water; leaving it recenters). An explicit --recenter-radius overrides with the
    # fixed-distance rule; a bare scene (no water geom) falls back to a 3 m radius.
    rec_bounds = None
    rec_radius = args.recenter_radius
    if observe and rec_radius is None:
        rec_bounds = _water_bounds(model)
        if rec_bounds is None:
            rec_radius = 3.0

    def key_cb(keycode):
        if 32 <= keycode < 127:                   # printable -> letter keys
            teleop.on_key(chr(keycode))

    if observe:
        if not args.recenter:
            where = "auto-recenter OFF — it may drift out of view"
        elif rec_bounds is not None:
            where = (f"auto-recenter when it leaves the water "
                     f"(±{rec_bounds[2]:.1f} x ±{rec_bounds[3]:.1f} m, "
                     f"z {rec_bounds[4]:.2f}..{rec_bounds[5]:.2f} m)")
        else:
            where = f"auto-recenter at {rec_radius:.1f} m drift"
        print("OBSERVE (free drift): the ROV is UNCONTROLLED — released from rest and "
              "carried by current+waves. Drive keys OFF; G toggles disturbances, H recenters"
              + (", N re-draws random headings" if getattr(teleop, "_dir_random", False) else "")
              + f"; {where}.")
    else:
        print("Keys come from the FOCUSED VIEWER WINDOW — click the viewer, then drive.")
    if teleop.show_arrows:
        print("Force arrows ON (press V to hide).  Legend:  " + LEGEND)
        print(f"  length = magnitude (force {FORCE_SCALE} m/N cap {FORCE_CAP} m; "
              f"velocity {VEL_SCALE} m per m/s cap {VEL_CAP} m).")
    else:
        print("Force arrows OFF (press V to show).")
    last = [0.0]
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        if observe:                               # frame the drift region from the side
            viewer.cam.lookat[:] = home[0][:3]
            span = (max(rec_bounds[2], rec_bounds[3]) if rec_bounds is not None
                    else (rec_radius if rec_radius is not None else 1.2))
            viewer.cam.distance = max(3.0, 2.0 * span + 1.5)
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
        while viewer.is_running():
            t0 = time.time()
            for _ in range(n_sub):
                if mission is not None:
                    mission.step(model, data)
                elif controller is not None:
                    controller.apply(model, data)
                mujoco.mj_step(model, data)
            if observe and (args.recenter or teleop._recenter_request):
                _maybe_recenter(model, data, hydro, home, rec_radius,
                                force=teleop._recenter_request, bounds=rec_bounds)
                teleop._recenter_request = False
            if observe and teleop._redraw_request and teleop._dir_apply is not None:
                b, t = teleop._dir_draw()
                teleop._dir_apply(b, t)       # mutate wave/current headings in place
                teleop._redraw_request = False
                print(f"\n[observe] new headings: wave beta={b:.0f}°, current theta_c={t:.0f}°")
            if teleop.show_arrows:
                mags = draw_force_arrows(viewer.user_scn, hydro, teleop, data, bid)
            else:                                     # arrows hidden (V) -> clear + just read mags
                viewer.user_scn.ngeom = 0
                mags = force_magnitudes(hydro, teleop, data, bid)
            if mission is not None:                   # overlay the planned square
                _draw_plan(viewer.user_scn, mission.plan_points())
            if surf is not None:                      # animate the water (VISUAL ONLY)
                field = teleop.disturbance
                surf.update(field, data.time,
                            enabled=(field is not None and getattr(field, "enabled", False)),
                            viewer=viewer)
            viewer.sync()
            if args.plot:
                for k in hist:
                    hist[k].append(mags.get(k, 0.0))
            if t0 - last[0] > 0.12:
                if observe:
                    _status_observe(teleop, data, home, mags)
                else:
                    _status(teleop, mags, hist if args.plot else None)
                last[0] = t0
            if monitor is not None and t0 - last_mon[0] > 0.033:   # ~30 Hz
                s = monitor_sample(hydro, data, bid)
                s["rec"] = mission.rec.active if mission is not None else False
                monitor.push(s)
                last_mon[0] = t0
            slack = n_sub * dt - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)
    print("\nbye")


def run_managed(model, data, teleop, hydro, args, monitor=None, controller=None,
                mission=None):
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
    if monitor is not None:
        # launch() owns the main thread (no per-frame hook), so feed the dashboard
        # from a read-only thread sampling live data/hydro (async to the viewer steps).
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        def _mon_loop():
            while not stop.is_set():
                monitor.push(monitor_sample(hydro, data, bid))
                time.sleep(0.033)
        threading.Thread(target=_mon_loop, daemon=True).start()
    # NOTE: autonomous control (--goto-origin/--square) is NOT driven here — it would
    # race the viewer's physics thread on MjData (see the guard in main()). Those modes
    # require the passive (default) or --viser viewer, which step + control on one thread.
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
# viser remote mode (headless: NO GLFW; browser UI over SSH/Tailscale)
# ---------------------------------------------------------------------------
def _mat2wxyz(xmat):
    """MuJoCo world rotation matrix (flat 9,) -> viser wxyz quaternion."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, dtype=float).reshape(9))
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _geom_color(model, gid):
    """(rgb 0-255, alpha) for a geom: its material, or its own rgba if set."""
    matid = int(model.geom_matid[gid])
    grgba = np.asarray(model.geom_rgba[gid], float)
    rgba = grgba
    if matid >= 0 and np.allclose(grgba, [0.5, 0.5, 0.5, 1.0]):
        rgba = np.asarray(model.mat_rgba[matid], float)   # default geom rgba -> use material
    rgb = tuple(int(np.clip(c, 0, 1) * 255) for c in rgba[:3])
    return rgb, float(rgba[3])


def _mesh_verts_faces(model, gid):
    """Vertices/faces of a mesh geom, in the geom's local frame."""
    mid = int(model.geom_dataid[gid])
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    verts = np.asarray(model.mesh_vert[va:va + vn]).reshape(-1, 3).astype(np.float32)
    faces = np.asarray(model.mesh_face[fa:fa + fn]).reshape(-1, 3).astype(np.uint32)
    return verts, faces


def build_viser_scene(server, model, data):
    """Create one viser node per visible MuJoCo geom. Returns [(gid, handle), ...];
    each frame we set handle.position/.wxyz from data.geom_xpos/geom_xmat (this is
    the mjModel+mjData -> Viser scene synchronization)."""
    mujoco.mj_forward(model, data)            # populate geom_xpos / geom_xmat
    G = mujoco.mjtGeom
    handles = []
    for gid in range(model.ngeom):
        rgb, alpha = _geom_color(model, gid)
        if alpha <= 0.02:                     # skip invisible (e.g. collision) geoms
            continue
        opacity = None if alpha >= 0.999 else alpha
        gtype = int(model.geom_type[gid])
        size = np.asarray(model.geom_size[gid], float)
        name = f"/mj/geom_{gid}"
        try:
            if gtype == G.mjGEOM_MESH:
                v, f = _mesh_verts_faces(model, gid)
                h = server.scene.add_mesh_simple(name, v, f, color=rgb,
                                                 opacity=opacity, flat_shading=False)
            elif gtype == G.mjGEOM_BOX:
                h = server.scene.add_box(name, color=rgb,
                                         dimensions=tuple(2.0 * size), opacity=opacity)
            elif gtype == G.mjGEOM_SPHERE:
                h = server.scene.add_icosphere(name, radius=float(size[0]),
                                               color=rgb, opacity=opacity)
            elif gtype == G.mjGEOM_ELLIPSOID:
                h = server.scene.add_icosphere(name, radius=1.0, color=rgb,
                                               scale=tuple(size), opacity=opacity)
            elif gtype in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
                import trimesh
                if gtype == G.mjGEOM_CAPSULE:
                    tm = trimesh.creation.capsule(height=2 * float(size[1]),
                                                  radius=float(size[0]))
                else:
                    tm = trimesh.creation.cylinder(radius=float(size[0]),
                                                   height=2 * float(size[1]))
                h = server.scene.add_mesh_simple(
                    name, np.asarray(tm.vertices, np.float32),
                    np.asarray(tm.faces, np.uint32), color=rgb, opacity=opacity)
            elif gtype == G.mjGEOM_PLANE:
                continue                      # infinite floor; skip for a free body
            else:                             # hfield / sdf / unknown -> placeholder
                h = server.scene.add_icosphere(name, radius=float(max(size[0], 0.05)),
                                               color=rgb, opacity=opacity)
        except Exception as e:
            print(f"[viser] geom {gid} (type {gtype}) skipped: {e}")
            continue
        h.position = tuple(map(float, data.geom_xpos[gid]))
        h.wxyz = _mat2wxyz(data.geom_xmat[gid])
        handles.append((gid, h))
    return handles


def viser_draw_arrows(server, hydro, teleop, data, bid):
    """Draw the same force/flow arrows as the local viewer, in viser. Re-adding
    with the same name overwrites the previous arrows (viser scene-tree semantics)."""
    starts, ends, cols = [], [], []
    for _name, pt, vec, scale, color, cap, min_mag in force_items(hydro, teleop, data, bid):
        mag = float(np.linalg.norm(vec))
        if mag <= min_mag:
            continue
        pt = np.asarray(pt, float)
        starts.append(pt)
        ends.append(pt + (np.asarray(vec, float) / mag) * min(scale * mag, cap))
        cols.append([int(np.clip(c, 0, 1) * 255) for c in color[:3]])
    if not starts:                            # nothing active -> hide the arrows node
        server.scene.add_arrows("/mj/forces", np.zeros((1, 2, 3)),
                                (0, 0, 0), visible=False)
        return
    points = np.stack([np.asarray(starts), np.asarray(ends)], axis=1)  # (N, 2, 3)
    server.scene.add_arrows("/mj/forces", points, np.asarray(cols, np.uint8),
                            shaft_radius=0.008, head_radius=0.02, head_length=0.04)


def _status_text(teleop, mags):
    w = teleop.wrench
    return (f"cmd   surge {w[0]:+.0f}  sway {w[1]:+.0f}  heave {w[2]:+.0f}  "
            f"yaw {w[5]:+.0f}  roll {w[3]:+.0f}\n"
            f"F(N)  thrust {mags.get('thrust', 0):.0f}  buoy {mags.get('buoyancy', 0):.0f}  "
            f"drag {mags.get('drag', 0):.0f}  kick {mags.get('kick', 0):.0f}\n"
            f"flow  cur {mags.get('current', 0):.2f}  wav {mags.get('wave', 0):.2f} m/s")


def build_viser_gui(server, model, data, teleop, hydro, args, recorder=None):
    """The browser control panel: buttons/sliders mapped onto the SAME
    Teleop.on_key path the keyboard uses (so both backends drive identically).
    Returns (status_handle, rec_status_handle) to refresh in the loop."""
    field = teleop.disturbance
    server.gui.add_markdown(
        "**BlueROV2 teleop (FLU).** Pitch is underactuated and never commanded; "
        "buttons latch a body wrench until changed.")

    def _buttons(folder, items):
        with server.gui.add_folder(folder):
            for label, key in items:
                server.gui.add_button(label).on_click(
                    lambda _evt, k=key: teleop.on_key(k))

    observe = getattr(teleop, "observe", False)
    if not observe:                          # drive controls — hidden in observe (uncontrolled)
        _buttons("Translate", [("Surge +  (W, forward)", "W"), ("Surge −  (S, back)", "S"),
                               ("Sway +  (Q, PORT/left)", "Q"), ("Sway −  (E, right)", "E"),
                               ("Heave +  (R, up)", "R"), ("Heave −  (F, down)", "F")])
        _buttons("Rotate", [("Yaw +  (A, CCW)", "A"), ("Yaw −  (D, CW)", "D"),
                            ("Roll +  (Z)", "Z"), ("Roll −  (C)", "C")])
        server.gui.add_button("STOP  (X)  — zero thrust", color="red").on_click(
            lambda _evt: teleop.on_key("X"))

    if field is not None:
        dist = server.gui.add_checkbox(
            "Disturbances: current + waves + kicks (G)", bool(field.enabled))
        dist.on_update(lambda _evt: setattr(field, "enabled", bool(dist.value)))

    arrows = server.gui.add_checkbox("Force/flow arrows (V)", bool(getattr(teleop, "show_arrows", True)))
    arrows.on_update(lambda _evt: setattr(teleop, "show_arrows", bool(arrows.value)))

    if observe:                              # no drive buttons — just recenter + re-draw
        server.gui.add_markdown(
            "**Observe (free drift).** The ROV is UNCONTROLLED — the current + waves carry "
            "it. Use Recenter to bring it back into view.")
        server.gui.add_button("⟲ Recenter drift (H)").on_click(
            lambda _evt: setattr(teleop, "_recenter_request", True))
        if getattr(teleop, "_dir_random", False):
            server.gui.add_button("🎲 New random headings (N)").on_click(
                lambda _evt: setattr(teleop, "_redraw_request", True))

    scale = server.gui.add_slider("Command scale", min=0.2, max=3.0, step=0.1,
                                  initial_value=float(args.scale))
    scale.on_update(lambda _evt: setattr(teleop, "keymap", keymap(scale.value)))

    def _reset(_evt):
        mujoco.mj_resetData(model, data)
        if hydro is not None:
            hydro.reset()
        with teleop.lock:
            teleop.wrench[:] = 0.0
        T.set_thruster_forces(model, data, np.zeros(6))
    server.gui.add_button("Reset pose").on_click(_reset)

    rec_status = None
    if recorder is not None:
        with server.gui.add_folder("Recording (CSV)"):
            rec_status = server.gui.add_text("rec", initial_value="idle", disabled=True)

            def _rec_start(_evt):
                try:
                    p = recorder.start()
                except Exception as e:
                    rec_status.value = f"record FAILED: {e}"
                    return
                rec_status.value = f"REC -> {os.path.basename(p)}"

            def _rec_stop(_evt):
                p = recorder.stop()
                rec_status.value = (f"saved {os.path.basename(p)} ({recorder.n} rows)"
                                    if p else "idle")
            server.gui.add_button("● Record", color="green").on_click(_rec_start)
            server.gui.add_button("■ Stop", color="red").on_click(_rec_stop)

    status = server.gui.add_text("status", initial_value="", multiline=True, disabled=True)
    return status, rec_status


class _ViserMonitor:
    """Browser-side monitor for `--viser --monitor` (remote): a water-velocity
    time-series uplot (NUMERIC seconds, not a date axis) and THREE separate image
    panels -- (x,y), (x,z), (y,z) -- each a scatter coloured by time (viridis, old->now)
    with the planned trajectory overlaid, plus the time-coloured 3D trajectory in the
    scene. uplot can't colour points by time, so the projections are rendered off-screen
    with matplotlib (Agg) and pushed as images. The time origin is the first sample and
    RESETS to 0 when recording starts. push() is cheap; refresh()/image render are
    throttled by the caller."""

    def __init__(self, server, window_s=30.0, plan=None):
        self.server = server
        self.window_s = float(window_s)
        self.plan = None if plan is None else np.asarray(plan, float)
        n = int(self.window_s * 60) + 64
        self.t = collections.deque(maxlen=n)
        self.vx = collections.deque(maxlen=n)
        self.vy = collections.deque(maxlen=n)
        self.vz = collections.deque(maxlen=n)
        self.px = collections.deque(maxlen=n)
        self.py = collections.deque(maxlen=n)
        self.pz = collections.deque(maxlen=n)
        self.t0 = None                           # time axis origin (first sample / rec start)
        self._rec = False                        # last-seen recording state (edge detect)
        self._img_n = 0
        # 256-entry viridis LUT for the time-colored 3D trajectory (gray fallback)
        try:
            from matplotlib import colormaps
            self._lut = (colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        except Exception:
            g = np.linspace(40, 255, 256).astype(np.uint8)
            self._lut = np.stack([g, g, g], axis=1)
        # one reusable off-screen panel, rendered once per projection per refresh
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self._fig = Figure(figsize=(4.6, 3.8), dpi=80, facecolor="#0e0e0e")
        self._canvas = FigureCanvasAgg(self._fig)
        self._ax = self._fig.add_subplot(1, 1, 1)
        z = np.zeros(1); e = np.zeros(0)
        with server.gui.add_folder("Monitor (water vel + path)"):
            self.vel = server.gui.add_uplot(
                (z, z, z, z),
                ({"label": "t (s)"},
                 {"label": "Vx", "stroke": "#ef5350", "width": 2},
                 {"label": "Vy", "stroke": "#66bb6a", "width": 2},
                 {"label": "Vz", "stroke": "#64b5f6", "width": 2}),
                scales={"x": {"time": False}},        # NUMERIC seconds, not a date axis
                title="Water velocity (m/s) vs time (s)", aspect=2.0)
            self.img_xy = server.gui.add_image(self._render_one(e, e, "x", "y", 0, 1, True, e),
                                               label="path (x, y) — top, colour = time", format="jpeg")
            self.img_xz = server.gui.add_image(self._render_one(e, e, "x", "z", 0, 2, False, e),
                                               label="path (x, z) — side, colour = time", format="jpeg")
            self.img_yz = server.gui.add_image(self._render_one(e, e, "y", "z", 1, 2, False, e),
                                               label="path (y, z) — colour = time", format="jpeg")

    def push(self, t, vtot, pos, recording=False):
        # restart the clock at 0 when recording STARTS, or when sim time jumps back (reset)
        reset = (recording and not self._rec) or (self.t and t < self.t[-1])
        self._rec = recording
        if reset:
            for d in (self.t, self.vx, self.vy, self.vz, self.px, self.py, self.pz):
                d.clear()
            self.t0 = None
        if self.t0 is None:
            self.t0 = t
        self.t.append(t)
        self.vx.append(vtot[0]); self.vy.append(vtot[1]); self.vz.append(vtot[2])
        self.px.append(pos[0]); self.py.append(pos[1]); self.pz.append(pos[2])

    def refresh(self):
        if not self.t:
            return
        t = np.fromiter(self.t, float)
        i0 = int(np.searchsorted(t, t[-1] - self.window_s))
        tt = t[i0:]
        vx = np.fromiter(self.vx, float)[i0:]
        vy = np.fromiter(self.vy, float)[i0:]
        vz = np.fromiter(self.vz, float)[i0:]
        px = np.fromiter(self.px, float)[i0:]
        py = np.fromiter(self.py, float)[i0:]
        pz = np.fromiter(self.pz, float)[i0:]
        self.vel.data = (tt - self.t0, vx, vy, vz)    # numeric seconds from 0
        span = (tt[-1] - tt[0]) if tt.size >= 2 else 0.0
        norm = np.clip((tt - tt[0]) / span, 0.0, 1.0) if span > 0 else np.zeros(tt.size)
        # time-colored 3D trajectory (viridis: old -> now) in the same 3D scene
        colors = self._lut[(norm * 255).astype(int)]
        pts3 = np.column_stack([px, py, pz]).astype(np.float32)
        self.server.scene.add_point_cloud("/monitor/traj", pts3, colors,
                                          point_size=0.015, point_shape="circle")
        # the three projection scatters (render is heavier -> throttle further)
        self._img_n += 1
        if self._img_n % 6 == 1:
            self.img_xy.image = self._render_one(px, py, "x", "y", 0, 1, True, norm)
            self.img_xz.image = self._render_one(px, pz, "x", "z", 0, 2, False, norm)
            self.img_yz.image = self._render_one(py, pz, "y", "z", 1, 2, False, norm)

    def _render_one(self, a, b, la, lb, ia, ib, equal, norm):
        """Render ONE time-coloured projection (+ planned trajectory) to an RGB array.
        (x,y) keeps equal aspect; (x,z)/(y,z) autoscale so small depth variation shows."""
        ax = self._ax
        ax.clear()
        ax.set_facecolor("#0e0e0e")
        if self.plan is not None:
            if len(self.plan) > 1:
                ax.plot(self.plan[:, ia], self.plan[:, ib], "--", color="#ffb300",
                        lw=1.5, zorder=1, label="planned")
            else:
                ax.scatter(self.plan[:, ia], self.plan[:, ib], marker="+",
                           color="#ffb300", s=120, zorder=1, label="target")
        if a.size:
            ax.scatter(a, b, c=norm, cmap="viridis", s=11, zorder=2)
        ax.set_xlabel(f"{la} (m)", color="#dddddd", fontsize=11)
        ax.set_ylabel(f"{lb} (m)", color="#dddddd", fontsize=11)
        if equal:
            ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.25)
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#444444")
        self._fig.tight_layout()
        self._canvas.draw()
        return np.asarray(self._canvas.buffer_rgba())[:, :, :3].copy()


def run_viser(model, data, teleop, hydro, args, controller=None, mission=None):
    """Headless remote mode: NO GLFW. Serve the scene + GUI over HTTP via viser;
    physics / hydro / disturbances run exactly as in the local modes."""
    import viser

    server = viser.ViserServer(host="0.0.0.0", port=args.port, verbose=False)
    try:
        server.scene.set_up_direction("+z")    # MuJoCo FLU is z-up
    except Exception:
        pass
    server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.008)
    handles = build_viser_scene(server, model, data)
    from recorder import Recorder, record_row
    # In a --square mission the mission OWNS the recorder (auto start/stop); otherwise
    # offer the manual Record/Stop buttons.
    # CSV name = <trajectory>_<model> so it says both which path and which controller ran:
    #   --goto-origin --ctrl dobmpc -> origin_dobmpc_<ts>.csv ; plain teleop -> teleop_<ts>.csv
    _tag = f"origin_{controller.mode}" if controller is not None else "teleop"
    recorder = None if mission is not None else Recorder(os.path.join(HERE, "recordings"), tag=_tag)
    if recorder is not None:                                 # attach sidecar JSON manifest
        _traj = (dict(kind="goto-origin", setpoint=list(getattr(controller, "p_ref", [0, 0, 0])),
                      yaw_ref=float(getattr(controller, "yaw_ref", 0.0)))
                 if controller is not None else dict(kind="teleop"))
        recorder.set_meta(lambda: _run_manifest(teleop.disturbance, controller, args, model, _traj))
    status, rec_status = build_viser_gui(server, model, data, teleop, hydro, args, recorder)
    mstatus = (server.gui.add_text("mission", initial_value="", disabled=True)
               if mission is not None else None)
    if mission is not None:                          # preview the planned square path
        pts = mission.plan_points()
        segs = np.stack([pts[:-1], pts[1:]], axis=1)              # (4, 2, 3) edges
        server.scene.add_line_segments("/plan/square", segs, (255, 210, 80), line_width=3)
        server.scene.add_point_cloud("/plan/corners", pts[:-1].astype(np.float32),
                                     np.tile((255, 210, 80), (4, 1)).astype(np.uint8),
                                     point_size=0.02, point_shape="circle")
    vmon = None
    if args.monitor:                              # render the monitor in the browser
        try:
            vmon = _ViserMonitor(server, args.monitor_window,
                                 plan=_plan_points(mission, controller))
            print("[monitor] browser panels added (Monitor folder + 3D trajectory).")
        except Exception as e:
            print(f"[monitor] viser panels disabled ({e})")
            vmon = None

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    dt = model.opt.timestep
    n_sub = max(1, round((1.0 / 30.0) / dt))   # ~30 fps render over the network
    arrows_ok = True                               # flips off only if viser arrow drawing errors
    observe = getattr(teleop, "observe", False)
    home = (data.qpos.copy(), data.qvel.copy())    # release pose for --observe recenter
    rec_bounds = None                              # default boundary = the water volume
    rec_radius = args.recenter_radius
    if observe and rec_radius is None:
        rec_bounds = _water_bounds(model)
        if rec_bounds is None:
            rec_radius = 3.0                       # bare scene fallback
    last = [0.0]
    last_mon = [0.0]
    last_rec = [0.0]

    print(f"[viser] {len(handles)} geoms synced; serving on 0.0.0.0:{args.port} "
          f"(reachable over Tailscale/SSH).")
    print(f"Open  http://<machine-or-tailscale-ip>:{args.port}  in a browser and "
          f"drive from the GUI panel.  Ctrl-C here to quit.")
    try:
        while True:
            t0 = time.time()
            for _ in range(n_sub):
                if mission is not None:
                    mission.step(model, data)
                elif controller is not None:
                    controller.apply(model, data)
                mujoco.mj_step(model, data)     # hydro passive callback runs in here
            if observe and (args.recenter or teleop._recenter_request):
                _maybe_recenter(model, data, hydro, home, rec_radius,
                                force=teleop._recenter_request, bounds=rec_bounds)
                teleop._recenter_request = False
            if observe and teleop._redraw_request and teleop._dir_apply is not None:
                b, t = teleop._dir_draw()
                teleop._dir_apply(b, t)
                teleop._redraw_request = False
                print(f"\n[observe] new headings: wave beta={b:.0f}°, current theta_c={t:.0f}°")
            for gid, h in handles:              # mjData -> viser scene sync
                h.position = tuple(map(float, data.geom_xpos[gid]))
                h.wxyz = _mat2wxyz(data.geom_xmat[gid])
            if arrows_ok and teleop.show_arrows:
                try:
                    viser_draw_arrows(server, hydro, teleop, data, bid)
                except Exception as e:
                    print(f"\n[viser] force arrows disabled ({e})")
                    arrows_ok = False
            elif not teleop.show_arrows:            # toggled off -> hide the arrows node
                try:
                    server.scene.add_arrows("/mj/forces", np.zeros((1, 2, 3)),
                                            (0, 0, 0), visible=False)
                except Exception:
                    pass
            if t0 - last[0] > 0.15:
                mags = force_magnitudes(hydro, teleop, data, bid)
                status.value = _status_text(teleop, mags)
                _status(teleop, mags, None)     # also to the SSH terminal
                last[0] = t0
            if vmon is not None:
                s = monitor_sample(hydro, data, bid)
                rec = (mission.rec.active if mission is not None
                       else (recorder.active if recorder is not None else False))
                vmon.push(s["t"], s["vtot"], s["pos"], recording=rec)
                if t0 - last_mon[0] > 0.1:         # ~10 Hz network refresh
                    try:
                        vmon.refresh()
                    except Exception as e:
                        print(f"\n[monitor] viser panels disabled ({e})")
                        vmon = None
                    last_mon[0] = t0
            if mission is not None:                   # mission auto-records; show status
                if t0 - last_rec[0] > 0.3:
                    mstatus.value = mission.status()
                    last_rec[0] = t0
            elif recorder.active:                     # manual CSV logging (Record/Stop)
                recorder.log(record_row(data, bid, hydro))
                if rec_status is not None and t0 - last_rec[0] > 0.5:
                    rec_status.value = f"REC {recorder.n} rows -> {os.path.basename(recorder.path)}"
                    last_rec[0] = t0
            slack = n_sub * dt - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            saved = recorder.stop()                   # flush manual recording
            if saved:
                print(f"[record] saved {saved} ({recorder.n} rows)")
        try:
            server.stop()
        except Exception:
            pass
    print("\nbye")


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
    ap.add_argument("--viser", "--remote", dest="viser", action="store_true",
                    help="headless remote mode: serve a viser web UI (no GLFW) instead "
                         "of the local viewer; drive from browser GUI buttons/sliders")
    ap.add_argument("--port", type=int, default=8080,
                    help="(--viser) web server port (default 8080)")
    ap.add_argument("--no-arrows", action="store_true",
                    help="start with the force/flow arrow overlay OFF (all viewers). Toggle "
                         "live with the V key (local) or the checkbox (--viser).")
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
    ap.add_argument("--observe", "--drift", dest="observe", action="store_true",
                    help="observation mode: DON'T pilot the ROV — release it from rest and "
                         "watch the current+waves carry it (thrust stays 0, disturbances "
                         "forced ON, drive keys disabled). Press H to recenter; auto-recenters "
                         "when it drifts out of view. Best with POOL_TAGS=1 (animated water).")
    ap.add_argument("--recenter-radius", dest="recenter_radius", type=float, default=None,
                    help="(--observe) OPTIONAL fixed drift distance (m) that snaps the ROV back. "
                         "Default: no fixed radius — the ROV drifts freely anywhere INSIDE the "
                         "water volume and recenters only when it leaves the water boundary "
                         "(pool_water_surface extent; 3 m fallback if the scene has no water).")
    ap.add_argument("--no-recenter", dest="recenter", action="store_false", default=True,
                    help="(--observe) never recenter — let it drift freely out of view.")
    ap.add_argument("--random-dirs", dest="random_dirs", action="store_true",
                    help="draw the wave heading (beta) and current heading (theta_c) RANDOMLY "
                         "in [0,360). Default ON in --observe (press N to re-draw); use "
                         "--no-random-dirs to force the fixed +x defaults there.")
    ap.add_argument("--no-random-dirs", dest="no_random_dirs", action="store_true",
                    help="in --observe, keep fixed headings (wave/current both +x) instead of "
                         "the default random draw.")
    ap.add_argument("--wave-deg", dest="wave_deg", type=float, default=None,
                    help="pin the wave mean heading beta [deg, FLU]; overrides the random draw.")
    ap.add_argument("--current-deg", dest="current_deg", type=float, default=None,
                    help="pin the current heading theta_c [deg, FLU]; overrides the random draw.")
    ap.add_argument("--current-speed", dest="current_speed", type=float, default=None,
                    help="mean current speed |V_bar_c| [m/s] (default 0.20).")
    ap.add_argument("--dir-seed", dest="dir_seed", type=int, default=None,
                    help="seed the random heading draw for reproducibility (default: fresh "
                         "entropy each launch).")
    ap.add_argument("--monitor", dest="monitor", action="store_true", default=True,
                    help="live monitor (water velocity + ROV trajectory); ON by default. "
                         "Local viewers -> pyqtgraph window; --viser -> browser")
    ap.add_argument("--no-monitor", dest="monitor", action="store_false",
                    help="disable the live monitor (on by default)")
    ap.add_argument("--monitor-window", type=float, default=30.0,
                    help="(--monitor) rolling time window in seconds (default 30)")
    ap.add_argument("--goto-origin", dest="goto_origin", action="store_true",
                    help="autonomous baseline controller: drive to the global origin "
                         "(keyboard idle; G still toggles disturbances)")
    ap.add_argument("--ctrl", choices=("pd", "pid", "mpc", "dobmpc"), default="pid",
                    help="(--goto-origin) controller: pd/pid baseline, or mpc/dobmpc "
                         "(NMPC without/with the EAOB disturbance observer). Default pid.")
    ap.add_argument("--start", type=str, default="2,1.5,-1,45",
                    help="(--goto-origin/--square) initial pose 'x,y,z,yawdeg'")
    ap.add_argument("--square", action="store_true",
                    help="autonomous mission: approach origin, then auto-record while "
                         "tracking a square (x,y) trajectory N laps, then save the CSV")
    ap.add_argument("--laps", type=int, default=10, help="(--square) number of laps")
    ap.add_argument("--square-size", dest="square_size", type=float, default=1.0,
                    help="(--square) side length in m (default 1.0)")
    ap.add_argument("--square-speed", dest="square_speed", type=float, default=0.15,
                    help="(--square) path speed in m/s (default 0.15)")
    ap.add_argument("--waves", choices=("spectrum", "classic"), default="spectrum",
                    help="wave model: irregular JONSWAP spectrum (default) or the 3 "
                         "classic sinusoids")
    ap.add_argument("--sea", type=str, default="0.20,4.0",
                    help="(--waves spectrum) sea state 'Hs,Tp' (m,s; default 0.20,4.0)")
    ap.add_argument("--ideal-thrusters", dest="ideal_thrusters", action="store_true",
                    help="(--square/--goto-origin) use the ideal force path "
                         "(commanded==realized). Default: realistic T200 model "
                         "(deadband / fwd-rev asymmetry / motor lag / voltage).")
    ap.add_argument("--thruster-voltage", dest="thruster_voltage", type=float,
                    default=T.NOMINAL_VOLTAGE_SCALE,
                    help="(realistic thrusters) battery thrust scale vs the ~20 V base "
                         f"curve; default {T.NOMINAL_VOLTAGE_SCALE} = 4S nominal 14.8 V "
                         "(1.0=as-fitted ~20 V, 0.85=mild sag, 0.62=near-empty 13 V).")
    args = ap.parse_args()
    if args.managed and (args.goto_origin or args.square):
        # --managed cedes the main thread to mujoco.viewer.launch (its own physics
        # thread); driving the controller from another thread races MjData. The
        # passive (default) and --viser loops have a same-thread per-step hook.
        print("[error] --goto-origin / --square need the local viewer (default) or "
              "--viser; --managed has no thread-safe per-step control hook.")
        return

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)

    if args.selftest:
        selftest(model, data, scale=args.scale)
        return

    # --- disturbance headings: wave beta + current theta_c. Random by default in --observe
    # (fresh each launch; N re-draws), else fixed +x unless pinned via --wave-deg/--current-deg.
    dir_rng = np.random.default_rng(args.dir_seed)      # None -> OS entropy each launch
    random_dirs = args.random_dirs or (args.observe and not args.no_random_dirs)
    speed = args.current_speed if args.current_speed is not None else 0.20

    def _draw_dirs():
        b = (args.wave_deg if args.wave_deg is not None
             else float(dir_rng.uniform(0.0, 360.0)) if random_dirs else 0.0)
        t = (args.current_deg if args.current_deg is not None
             else float(dir_rng.uniform(0.0, 360.0)) if random_dirs else 0.0)
        return b, t

    beta0, theta0 = _draw_dirs()

    field = None
    if args.no_hydro:
        model.opt.gravity[:] = 0.0          # thruster-only mode (Phase 1/2 feel)
        hydro = None
    else:
        field = D.DisturbanceField(current=_current_from_heading(speed, theta0),
                                   waves=_waves_at_heading(args, beta0), seed=0)
        field.enabled = args.disturb
        hydro = H.Hydrodynamics(model, disturbance=field).install()  # gravity ON
    # --managed prints its status from on_key (terminal); passive prints from loop
    teleop = Teleop(model, data, scale=args.scale, verbose=args.managed,
                    disturbance=field)
    teleop.show_arrows = not args.no_arrows  # V toggles the force/flow arrow overlay live
    if field is not None:                    # re-drawable headings (N key / viser button)
        # "random" only if a re-draw could actually change something (>=1 heading unpinned)
        teleop._dir_random = random_dirs and (args.wave_deg is None or args.current_deg is None)
        teleop._beta, teleop._theta = beta0, theta0
        teleop._dir_draw = _draw_dirs

        def _apply(b, t):
            _apply_directions(field, args, b, t, speed)
            teleop._beta, teleop._theta = b, t
        teleop._dir_apply = _apply

    if args.observe:                        # free-drift observation mode
        if hydro is None or field is None:
            print("[error] --observe needs hydro + disturbances; remove --no-hydro.")
            return
        if args.goto_origin or args.square:
            print("[error] --observe is UNCONTROLLED free-drift; do not combine it with "
                  "--goto-origin / --square (those pilot the ROV).")
            return
        field.enabled = True                # the whole point: the flow acts from t=0
        teleop.observe = True               # gate drive keys; enable H recenter
        if not getattr(RM, "_POOL_TAGS", False):
            print("[observe] note: pool floor + water surface are OFF (you set POOL_TAGS=0). "
                  "teleop shows the pool by default — drop POOL_TAGS=0 to see the tags/water.")
        if args.managed:
            print("[observe] NOTE: --managed has no per-step hook, so it can't recenter; "
                  "it free-drifts and may leave view. Use the default viewer or --viser.")

    controller = None
    mission = None
    if args.goto_origin or args.square:
        from controller import PoseController
        sx, sy, sz, syaw = (float(v) for v in args.start.split(","))
        data.qpos[:3] = [sx, sy, sz]
        half = np.radians(syaw) / 2.0
        data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]   # yaw quaternion (wxyz)
        mujoco.mj_forward(model, data)
        # Realistic T200 actuator is ON by default for autonomous missions (they
        # exist to predict the real robot); --ideal-thrusters reverts to the
        # commanded==realized force path. Manual keyboard teleop is a separate path
        # (Teleop.on_key) and is unaffected.
        actuator = None if args.ideal_thrusters else T.ThrusterModel(
            n=RM.N_THRUSTERS, lag=True, voltage_scale=args.thruster_voltage)
        if args.ctrl in ("mpc", "dobmpc"):
            from dobmpc_controller import DOBMPCController
            controller = DOBMPCController(model, hydro=hydro, mode=args.ctrl,
                                          setpoint=(0.0, 0.0, 0.0), yaw_ref=0.0,
                                          actuator=actuator)
        else:
            controller = PoseController(model, mode=args.ctrl, setpoint=(0.0, 0.0, 0.0),
                                        yaw_ref=0.0, buoyancy_ff=hydro, actuator=actuator)
        if actuator is None:
            print("[thrusters] ideal force path (commanded == realized).")
        else:
            print(f"[thrusters] realistic T200: deadband + fwd/rev asymmetry + motor "
                  f"lag + voltage x{args.thruster_voltage:.2f} "
                  f"({'nominal 14.8 V' if abs(args.thruster_voltage - T.NOMINAL_VOLTAGE_SCALE) < 1e-6 else 'custom'}).")
        if args.square:
            from recorder import Recorder
            from mission import SquareMission
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            recorder = Recorder(os.path.join(HERE, "recordings"), tag=f"square_{args.ctrl}")
            recorder.set_meta(lambda: _run_manifest(            # sidecar JSON manifest
                field, controller, args, model,
                dict(kind="square", size=args.square_size, laps=args.laps,
                     speed=args.square_speed, depth=0.0)))
            mission = SquareMission(controller, recorder, hydro, bid,
                                    size=args.square_size, laps=args.laps,
                                    speed=args.square_speed)
            print(f"[square] {args.ctrl.upper()} {args.laps}-lap {args.square_size:.2f}m "
                  f"square @ {args.square_speed:.2f} m/s from "
                  f"({sx:+.1f},{sy:+.1f},{sz:+.1f}); auto-record on origin.")
        else:
            print(f"[goto-origin] {args.ctrl.upper()} controller -> origin; "
                  f"start ({sx:+.1f},{sy:+.1f},{sz:+.1f}) yaw {syaw:.0f}°. Keyboard idle.")

    print(HELP)
    print(("Hydro+gravity ON (in-water feel: drag stops you, buoyancy holds depth, "
           "tilt self-rights)." if hydro else
           "Hydro OFF, gravity OFF (thruster-only mode)."))
    if field is not None:
        print(f"Disturbances {'ON' if field.enabled else 'OFF'} (press G to toggle). "
              f"current |{np.linalg.norm(field.current):.2f}| m/s, waves, kicks; "
              f"depth ~{field.z_surface:.0f} m at start.")
        print(f"Headings: wave beta = {teleop._beta:.0f}°, current theta_c = "
              f"{teleop._theta:.0f}° (FLU)"
              + ("  [RANDOM — press N to re-draw]" if teleop._dir_random else "  [fixed]"))

    # --viser renders the monitor in the browser (see run_viser); the local viewers
    # use the separate-process pyqtgraph desktop window, which needs a display.
    monitor = None
    if args.monitor and not args.viser:
        try:
            from monitor import MonitorHandle
            monitor = MonitorHandle(window_s=args.monitor_window,
                                    plan=_plan_points(mission, controller))
            print("[monitor] dashboard process started.")
        except Exception as e:
            print(f"[monitor] disabled ({e})")
            monitor = None

    try:
        if args.viser:
            run_viser(model, data, teleop, hydro, args, controller=controller,
                      mission=mission)
        elif args.managed:
            run_managed(model, data, teleop, hydro, args, monitor=monitor,
                        controller=controller, mission=mission)
        else:
            run_passive(model, data, teleop, hydro, args, monitor=monitor,
                        controller=controller, mission=mission)
    finally:
        if monitor is not None:
            monitor.close()
        if mission is not None:
            mission.close()                 # flush/save any in-progress recording


if __name__ == "__main__":
    main()
