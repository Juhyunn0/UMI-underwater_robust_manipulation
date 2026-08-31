"""Phase-0 GPU/MJX verification (RTX 5090 / Blackwell, env `robust-mjx`).

Checks, in order:
  1. versions of jax / jaxlib / mujoco / mujoco-mjx / numpy
  2. jax.devices() -> confirm a CUDA device (the RTX 5090) is detected
  3. a tiny MJX rollout on a guaranteed-compatible synthetic model, run on GPU
  4. (bonus) attempt to load the canonical bluerov.xml under MJX and step it;
     failures here are reported as a model caveat, NOT a setup failure, because
     the CPU passive hydro callback is known not to run under MJX (see docs/06).

Run with the dedicated env's interpreter:
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/bdml/miniforge3/envs/robust-mjx/bin/python verify/verify_gpu_mjx.py
"""
import os
import sys

# Be a good citizen: a display server is using this GPU, so don't grab 75% of VRAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ok = True


def line():
    print("=" * 64)


# ----------------------------------------------------------------------
# 1. versions
# ----------------------------------------------------------------------
line()
print("1) VERSIONS")
import jax
import jaxlib
import numpy as np
import mujoco
from mujoco import mjx

print(f"   jax        {jax.__version__}")
print(f"   jaxlib     {jaxlib.__version__}")
print(f"   numpy      {np.__version__}")
print(f"   mujoco     {mujoco.__version__}")
try:
    import mujoco.mjx as _mjx_mod
    print(f"   mujoco-mjx (mjx submodule importable)")
except Exception as e:  # pragma: no cover
    print(f"   mujoco-mjx IMPORT FAILED: {e}")
    ok = False

# ----------------------------------------------------------------------
# 2. devices
# ----------------------------------------------------------------------
line()
print("2) JAX DEVICES")
print(f"   default backend : {jax.default_backend()}")
devs = jax.devices()
print(f"   jax.devices()   : {devs}")
gpu_devs = [d for d in devs if d.platform == "gpu"]
if gpu_devs:
    d = gpu_devs[0]
    desc = getattr(d, "device_kind", "?")
    print(f"   -> GPU detected : {d} (kind={desc})")
else:
    print("   -> NO GPU DEVICE DETECTED  (jax sees CPU only)")
    ok = False

# ----------------------------------------------------------------------
# 3. tiny MJX rollout on GPU (synthetic, guaranteed-compatible model)
# ----------------------------------------------------------------------
line()
print("3) TINY MJX ROLLOUT ON GPU (synthetic model)")
SYNTH_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
try:
    import jax.numpy as jnp
    m = mujoco.MjModel.from_xml_string(SYNTH_XML)
    mx = mjx.put_model(m)
    dx = mjx.make_data(mx)

    # Confirm the data actually lives on the GPU.
    qpos_dev = dx.qpos.devices()
    print(f"   data device     : {qpos_dev}")

    jit_step = jax.jit(mjx.step)
    # warm-up compile
    dx = jit_step(mx, dx)
    for _ in range(200):
        dx = jit_step(mx, dx)
    qpos = np.asarray(dx.qpos)
    z = float(qpos[2])
    finite = bool(np.all(np.isfinite(qpos)))
    on_gpu = any(getattr(dd, "platform", "") == "gpu" for dd in qpos_dev)
    print(f"   after 201 steps : z={z:+.4f} m (free-fall under gravity expected)")
    print(f"   all finite      : {finite}")
    print(f"   ran on GPU      : {on_gpu}")
    if not (finite and on_gpu and z < 0.99):
        ok = False
        print("   -> SYNTH ROLLOUT CHECK FAILED")
    else:
        print("   -> SYNTH MJX ROLLOUT ON GPU OK")
except Exception as e:
    ok = False
    print(f"   -> SYNTH MJX ROLLOUT FAILED: {type(e).__name__}: {e}")

# ----------------------------------------------------------------------
# 4. bonus: canonical bluerov.xml under MJX (caveat, not a gate)
# ----------------------------------------------------------------------
line()
print("4) CANONICAL bluerov.xml UNDER MJX (bonus; non-gating)")
xml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bluerov.xml")
try:
    mb = mujoco.MjModel.from_xml_path(xml_path)
    mbx = mjx.put_model(mb)
    dbx = mjx.make_data(mbx)
    jit_step = jax.jit(mjx.step)
    dbx = jit_step(mbx, dbx)        # compile + 1 step
    for _ in range(50):
        dbx = jit_step(mbx, dbx)
    qpos = np.asarray(dbx.qpos)
    finite = bool(np.all(np.isfinite(qpos)))
    print(f"   bluerov.xml loaded under MJX and stepped 51x on GPU")
    print(f"   qpos finite     : {finite}")
    print("   NOTE: CPU passive-callback hydro (hydro.py) does NOT run under MJX;")
    print("         re-expressing Fossen terms in JAX is a later phase (see docs/06).")
    if not finite:
        print("   -> bluerov MJX states not finite (investigate later)")
except Exception as e:
    print(f"   -> bluerov.xml NOT directly MJX-steppable yet: {type(e).__name__}: {e}")
    print("      (expected if the MJCF uses features MJX doesn't support; this is a")
    print("       model-porting note for a later phase, NOT a Phase-0 setup failure.)")

# ----------------------------------------------------------------------
line()
if ok:
    print("PHASE-0 GPU/MJX VERIFICATION PASSED  (5090 detected + MJX rollout on GPU)")
    sys.exit(0)
else:
    print("PHASE-0 GPU/MJX VERIFICATION FAILED")
    sys.exit(1)
