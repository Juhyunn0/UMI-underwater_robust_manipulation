# 05 — Keyboard teleop + live force-arrow visualization

**Status: DONE ✓.** Code: `teleop.py`. Drive the BlueROV2 in the MuJoCo viewer and
watch the external forces as live 3D arrows. Reuses `thrusters.py` (allocation),
`hydro.py` (forces), `disturbances.py` (current/waves/kicks). The viz only **reads**
the forces already computed by the physics — it does not change the dynamics.

## Controls (FLU) — unchanged from before

| key | action | | key | action |
|----|--------|---|----|--------|
| **W / S** | surge +x / −x (forward / back) | | **A / D** | yaw +Mz / −Mz (turn left / right) |
| **Q / E** | sway +y / −y (left / right, +y = PORT) | | **Z / C** | roll +Mx / −Mx |
| **R / F** | heave +z / −z (up / down) | | **X** | STOP (zero all thrust) |
| **G** | toggle disturbances (current+waves+kicks) | | Ctrl-C / close | quit |

Keys → fixed-magnitude **FLU body wrench** → `pinv(B)` → thruster forces (clamped to
[−51.55, 64.13] N) → `data.ctrl`. Only the **5 controllable DOFs** (pitch is never
commanded — underactuated, see [03_THRUSTERS.md](03_THRUSTERS.md)). Commands
**latch**. Gravity + hydro ON by default; surge kept gentle (`SURGE_N=8`) because of
the surge→pitch coupling; `--scale` pushes harder; `--no-hydro` = thruster-only.

## Force arrows (default mode)

Each frame, one color-coded arrow per external force is drawn on the vehicle via the
passive viewer's `user_scn`:

| color | force | from | notes |
|---|---|---|---|
| **green** | buoyancy / restoring | the CB (= COM + coBM·ẑ) | ~constant ~111 N up |
| **gray** | hydrodynamic drag (total damping) | COM | opposes relative motion |
| **orange** | net thrust | COM | matches your command |
| **red** | kick | COM | brief spike when a kick fires |
| **blue** | current (water velocity) | COM | steady flow arrow |
| **cyan** | wave (water velocity) | COM | oscillating flow arrow |

**Disturbance representation (documented choice):** current and waves act through
the *relative velocity* `vr = v − v_water` inside the drag/added-mass (so their force
is embedded in the **gray drag arrow**, not a separate force). They are therefore
shown as **flow arrows** (the water velocity itself) — blue current (steady), cyan
wave (oscillating) — i.e. the disturbance *source*. **Kicks are a real external
force**, shown as the red arrow.

**Scale:** arrow length = magnitude × scale, capped:
- forces: `0.003 m/N`, cap `0.6 m` (buoyancy 111 N → 0.33 m ≈ vehicle size);
- velocities: `0.5 m per m/s`, cap `0.4 m` (current 0.2 m/s → 0.10 m).
Arrows below ~0.5 N (or 0.01 m/s) are not drawn.

**Legend & magnitudes:** the color→force legend is printed to the console at
startup; a live console status line shows each force magnitude (and the command).
Each arrow also carries a `name + magnitude` label (e.g. "buoyancy 111N"), shown if
you enable labels in the viewer. `--plot` adds console sparklines of the drag / wave
/ kick magnitudes so the wave oscillation and kick spikes are visible over time.

## Two viewer modes

**Default — `launch_passive` (force arrows).** teleop runs its own step+sync loop
and updates `user_scn` each frame; keys arrive via the viewer's `key_callback`.
⚠ **Keys come from the FOCUSED VIEWER WINDOW** — click the viewer, then drive (this
is the opposite of `--managed`, where the terminal is focused).

**`--managed` — old managed viewer (no arrows).** `mujoco.viewer.launch` runs the
GUI on the main thread; a background keyboard thread reads the **terminal** and
writes `data.ctrl`. Plain `python` anywhere; handy for a quick macOS check without
mjpython. Managed `launch()` has no `key_callback`, which is why arrows need the
passive viewer. Backends: terminal cbreak (default) or `--pynput` (global capture).

## How to run (per platform)

```bash
cd bluerov2_mujoco_marinegym

# Ubuntu 22.04 (TARGET): launch_passive is Linux-native -> plain python, arrows
python teleop.py
python teleop.py --disturb --plot       # start disturbed, with sparklines

# macOS (temporary preview): launch_passive needs mjpython, and the project path
# has a space that breaks mjpython -> run from the no-space venv ~/bluerov_venv:
mjpython teleop.py                       # (after: python3 -m venv ~/bluerov_venv && ~/bluerov_venv/bin/pip install -U mujoco)

# macOS quick check WITHOUT mjpython (no arrows):
python teleop.py --managed

python teleop.py --selftest              # headless key->direction check (any platform)
```

`--selftest` (no display) asserts W→+x, S→−x, Q→+y, E→−y, R→+z, F→−z, A→+Mz, D→−Mz,
Z→+Mx, C→−Mx, X→zero — all pass. (Sway shows ~zero roll because the allocator cancels
the sway→roll coupling; surge keeps a small uncontrollable pitch — both consistent
with the rank-5 allocation.)

## Verification of the arrows

With disturbances on, the arrows match the physics (checked headlessly by drawing
into an `MjvScene`): buoyancy ~111 N up (≈ constant), drag opposes motion, thrust
matches the command, current arrow along the flow, wave arrow oscillates, and a kick
gives a brief red spike (e.g. ~31 N). The Phase 1–4 suites still pass — the viz does
not alter the dynamics.

## Notes / gotchas

- `launch_passive` is **Linux-native** (plain python). The macOS `mjpython` need is
  **temporary** (drafting only); on the RTX-5090 Ubuntu box `python teleop.py` works
  directly. See [06_ENVIRONMENT.md](06_ENVIRONMENT.md) for the space-in-path issue
  and the `~/bluerov_venv` workaround.
- Default mode = focus the **viewer**; `--managed` = focus the **terminal**.
- The viz reuses `hydro.components` / `hydro.water` (per-component force/velocity
  vectors exposed read-only) and the allocation `B` for net thrust — nothing in the
  physics path changed.
