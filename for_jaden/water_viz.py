"""Animated water surface for the pool scene — VISUAL ONLY, dynamics-inert.

Drives the `pool_water_surface` heightfield (built by tools/gen_pool_apriltags.py) so the
water visibly undulates like real waves and drifts with the current, reconstructed
from the SAME disturbance field the physics uses. Nothing here touches dynamics:
the hfield geom is contype=0 conaffinity=0, hydro acts only on base_link, and
model.hfield_data is written AFTER mj_step (never read back by mj_step for a
non-colliding geom). Verified delta=0 over a multi-thousand-step rollout.

Supports BOTH disturbance models via duck-typing on `field.waves`:
  * LEGACY  disturbances.py DisturbanceField (teleop): field.waves is a list of
    (U, omega, k, dir3, phase) tuples; elevation amplitude a = U/omega; current is
    the constant vector field.current.
  * MODERN  disturbance/env.py DisturbanceEnv (run_viewer): field.waves is a
    DirectionalWaveField exposing arrays a_m,k_m,ex_m,ey_m,omega_m,eps_m; current
    via field.current_velocity().

Surface elevation:  eta(x,y,t) = sum_i a_i cos(k_vis_i*(dir_i . (x,y)) - omega_i t + phase_i)
The whole pattern is advected by the integrated current so waves+current read as one
surface. Real ocean wavelengths (6-76 m) dwarf the pool (~1.8x4.9 m), so by default
(`lam_target=None`) the TRUE wavenumber is used (physically faithful: gentle heave/tilt);
set `lam_target` (m) to shrink the visual wavelength for exaggerated ripples. Neither
knob touches the physics field — this is a render-time reparameterization only.

Usage (interactive passive viewer):
    surf = make_surface(model)                 # None if the scene has no hfield water
    ...                                        # each frame, after mj_step, before sync:
    if surf: surf.update(field, data.time, enabled=field.enabled, viewer=viewer)

Usage (offscreen mujoco.Renderer):
    if surf: surf.update(field, data.time, enabled=field.enabled, renderer=renderer)
"""
from __future__ import annotations

import os
import numpy as np
import mujoco

WATER_GEOM = "pool_water_surface"
MAX_MODERN_COMPONENTS = 120   # cap modern spectrum for the visual (top-|a| components)


def make_surface_from_env(model, geom_name=WATER_GEOM):
    """make_surface() with the visual knobs read from the environment:
      WATER_WAVE_LAMBDA : visual wavelength in metres. Unset/0 -> physically faithful
                          (true ocean wavelengths; gentle heave on a small pool).
                          e.g. 0.9 for exaggerated, clearly-sloshing ripples.
      WATER_WAVE_AMP    : amplitude gain (default 1.0).
    Returns None if the scene has no hfield water surface."""
    lam = os.environ.get("WATER_WAVE_LAMBDA", "").strip()
    amp = os.environ.get("WATER_WAVE_AMP", "").strip()
    lam_target = float(lam) if (lam and float(lam) > 0) else None
    amp_gain = float(amp) if amp else 1.0
    return make_surface(model, geom_name=geom_name,
                        lam_target=lam_target, amp_gain=amp_gain)


def make_surface(model, geom_name=WATER_GEOM, **kw):
    """Return a WaterSurface if `geom_name` exists and is a heightfield, else None.
    Lets callers stay agnostic: only the animated POOL_TAGS scene has the hfield."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0 or int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
        return None
    return WaterSurface(model, geom_name=geom_name, **kw)


class WaterSurface:
    def __init__(self, model, geom_name=WATER_GEOM,
                 lam_target=None, amp_gain=1.0, max_components=MAX_MODERN_COMPONENTS):
        self.model = model
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid < 0 or int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
            raise ValueError(f"geom {geom_name!r} is not a heightfield")
        self.hfid = int(model.geom_dataid[gid])
        self.nrow = int(model.hfield_nrow[self.hfid])          # spans X (radius_x)
        self.ncol = int(model.hfield_ncol[self.hfid])          # spans Y (radius_y)
        hx, hy, self.elev, _base = (float(v) for v in model.hfield_size[self.hfid])
        cx, cy, _cz = (float(v) for v in model.geom_pos[gid])
        # world (x,y) at each hfield cell centre; row-major data[r*ncol + c], row=X col=Y
        r = (np.arange(self.nrow) + 0.5) / self.nrow
        c = (np.arange(self.ncol) + 0.5) / self.ncol
        X = cx + (-hx + r * 2.0 * hx)
        Y = cy + (-hy + c * 2.0 * hy)
        self.XX, self.YY = np.meshgrid(X, Y, indexing="ij")    # (nrow, ncol)
        adr = int(model.hfield_adr[self.hfid])
        self._slice = slice(adr, adr + self.nrow * self.ncol)
        self.lam_target = lam_target
        self.amp_gain = float(amp_gain)
        self.max_components = int(max_components)
        self._flat = np.full((self.nrow, self.ncol), 0.5, np.float32)
        # current advection accumulator (integrated drift), and per-field subset cache
        self.Xc = self.Yc = 0.0
        self._t_prev = None
        self._sub_key = None
        self._sub = None

    # ---- public --------------------------------------------------------------
    def update(self, field, t, enabled=True, viewer=None, renderer=None):
        """Recompute the surface for time `t` and push it to the GPU.
        viewer   : passive-viewer Handle -> viewer.update_hfield (public, preferred).
        renderer : offscreen mujoco.Renderer -> mjr_uploadHField via its context.
        Returns True if the GPU upload was issued."""
        if enabled and field is not None:
            eta = self._eta(field, t)
            d = 0.5 + eta / max(self.elev, 1e-6)
            data = np.clip(d, 0.0, 1.0).astype(np.float32)
        else:
            data = self._flat
        self.model.hfield_data[self._slice] = data.ravel()
        return self._upload(viewer, renderer)

    # ---- elevation -----------------------------------------------------------
    def _eta(self, field, t):
        Xa, Ya = self._advected(field, t)
        wf = getattr(field, "waves", None)
        if hasattr(wf, "a_m"):                         # MODERN DirectionalWaveField
            return self._eta_modern(wf, Xa, Ya, t)
        return self._eta_legacy(field, Xa, Ya, t)      # LEGACY tuple list

    def _eta_legacy(self, field, Xa, Ya, t):
        eta = np.zeros_like(Xa)
        for comp in (getattr(field, "waves", None) or []):
            U, omega, k, dir3, phase = comp
            kv = self._scale_k(np.asarray(k, float))
            a = (U / max(float(omega), 1e-6)) * self.amp_gain
            eta = eta + a * np.cos(kv * dir3[0] * Xa + kv * dir3[1] * Ya
                                   - float(omega) * t + float(phase))
        return eta

    def _eta_modern(self, wf, Xa, Ya, t):
        a, k, ex, ey, omega, eps = self._modern_subset(wf)
        kv = self._scale_k(k)
        # (nrow, ncol, M)
        theta = (Xa[..., None] * (kv * ex) + Ya[..., None] * (kv * ey)
                 - omega * t + eps)
        return (a * self.amp_gain * np.cos(theta)).sum(axis=-1)

    def _modern_subset(self, wf):
        """Cache the top-|a| components of a modern wave field (visual-only trim)."""
        key = id(wf)
        if self._sub_key != key:
            a = np.asarray(wf.a_m, float)
            idx = np.arange(a.size)
            if a.size > self.max_components:
                idx = np.argsort(a)[-self.max_components:]
            self._sub = (a[idx], np.asarray(wf.k_m, float)[idx],
                         np.asarray(wf.ex_m, float)[idx], np.asarray(wf.ey_m, float)[idx],
                         np.asarray(wf.omega_m, float)[idx], np.asarray(wf.eps_m, float)[idx])
            self._sub_key = key
        return self._sub

    def _scale_k(self, k):
        if self.lam_target is None:
            return k                                   # faithful: true wavenumber
        return np.full_like(np.asarray(k, float), 2.0 * np.pi / float(self.lam_target))

    def _advected(self, field, t):
        vx, vy = _current_vec(field)[:2]
        if self._t_prev is not None:
            dt = float(t) - self._t_prev
            if 0.0 <= dt < 1.0:                        # ignore resets / big jumps
                self.Xc += vx * dt
                self.Yc += vy * dt
        self._t_prev = float(t)
        return self.XX - self.Xc, self.YY - self.Yc    # sample-shift == pattern drift +current

    # ---- gpu upload ----------------------------------------------------------
    def _upload(self, viewer, renderer):
        if viewer is not None:
            viewer.update_hfield(self.hfid)            # public, passive viewer
            return True
        if renderer is not None:
            ctx = getattr(renderer, "_mjr_context", None)
            if ctx is not None:
                try:
                    mujoco.mjr_uploadHField(self.model, ctx, self.hfid)
                    return True
                except Exception:
                    return False                       # caller falls back to static
        return False


def _current_vec(field):
    """Horizontal current vector [vx,vy,vz] (FLU) from either disturbance field."""
    wf = getattr(field, "waves", None)
    if hasattr(wf, "a_m"):                             # MODERN env
        for call in (lambda: field.current_velocity(field._t_last),
                     lambda: field.current_velocity()):
            try:
                return np.asarray(call(), float)
            except (TypeError, AttributeError):
                continue
            except Exception:
                break
        cur = getattr(field, "current", None)
        if cur is not None and hasattr(cur, "mean_velocity"):
            try:
                return np.asarray(cur.mean_velocity(), float)
            except Exception:
                pass
        return np.zeros(3)
    cur = getattr(field, "current", None)              # LEGACY constant vector
    return np.asarray(cur, float) if cur is not None else np.zeros(3)
