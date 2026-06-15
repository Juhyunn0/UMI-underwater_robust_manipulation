#!/usr/bin/env python3
"""
Environmental disturbances for the BlueROV2 sim (Phase 4), FLU frame.

Three layers, plus a domain-randomization sampler:

  1. Uniform current  — a constant FLU water-velocity vector. It enters the
     physics as a WATER VELOCITY: hydro (hydro.py) uses the relative velocity
     vr = v - v_water in drag / Coriolis / added-mass, so an unpowered vehicle is
     carried by the flow (Fossen-correct) and station-keeping needs thrust.
  2. Waves            — a few sinusoidal water-velocity components (deep-water
     orbital motion) whose amplitude DECAYS with depth as exp(-k*depth), with
     k = omega^2 / g, so longer-period swell penetrates deeper. Also fed through
     v_water, so wave drag AND wave added-mass excitation both come for free.
  3. Random kicks     — Poisson-timed impulsive force spikes (random direction +
     magnitude), applied directly as an external world-frame force (a "bump").

Domain randomization: `sample_config(seed)` / `randomize(seed)` draw a fresh,
bounded config (all disturbance params + a few model params: drag scale, thruster
scale, buoyancy trim) for a future episode reset.

Everything is FLU (z up). Water velocity is a world-frame FLU vector; hydro does
the world->body rotation. Only numpy required.

Depth convention: depth d = max(0, z_surface - z_body). Default z_surface = 3.0 m,
so the body at the model origin (z=0) sits at ~3 m depth (the target site).
"""
import numpy as np

G = 9.81

# ---- reasonable defaults for a ~3 m-deep site (see the note / docs) ----------
DEFAULT_CURRENT = (0.20, 0.0, 0.0)          # m/s, mostly horizontal (FLU)
DEFAULT_WAVES = [                            # U = surface orbital speed (m/s)
    dict(U=0.18, T=7.0, heading_deg=0.0,   phase_deg=0.0),    # long swell (deep-reaching)
    dict(U=0.12, T=3.5, heading_deg=50.0,  phase_deg=90.0),   # wind wave
    dict(U=0.08, T=2.0, heading_deg=-25.0, phase_deg=200.0),  # short chop (decays fast)
]
DEFAULT_KICKS = dict(rate=0.2, fmin=20.0, fmax=50.0, duration=0.15)  # ~1 / 5 s, 20-50 N
DEFAULT_Z_SURFACE = 3.0

# ---- documented domain-randomization ranges ----------------------------------
DR_RANGES = dict(
    current_speed=(0.0, 0.4),          # m/s
    current_vertical=(-0.03, 0.03),    # m/s (small)
    n_waves=(1, 3),
    wave_U=(0.05, 0.25),               # m/s surface orbital speed
    wave_T=(2.0, 9.0),                 # s  (incl. long swell)
    kick_rate=(0.1, 0.5),              # events / s
    kick_fmin=(8.0, 20.0),             # N
    kick_fmax=(30.0, 60.0),            # N
    kick_duration=(0.10, 0.20),        # s
    drag_scale=(0.7, 1.3),             # multiplies D_L and D_NL
    thruster_scale=(0.8, 1.2),         # multiplies commanded thrust
    buoyancy_trim=(-2.0, 2.0),         # N added to net buoyancy
)


def _heading3(deg):
    a = np.radians(deg)
    return np.array([np.cos(a), np.sin(a), 0.0])


# ---- irregular waves: JONSWAP spectrum (random-phase linear superposition) -----
DEFAULT_HS = 0.20      # significant wave height [m]   (small coastal/sheltered site)
DEFAULT_TP = 4.0       # peak period [s]
DEFAULT_GAMMA = 3.3    # JONSWAP peak factor (gamma=1 -> Pierson-Moskowitz)
DEFAULT_NWAVE = 30     # number of components
DEFAULT_SPREAD_S = 4.0 # cos^(2s) directional spreading (larger = narrower)


def jonswap_wave_specs(Hs=DEFAULT_HS, Tp=DEFAULT_TP, n=DEFAULT_NWAVE, gamma=DEFAULT_GAMMA,
                       heading_deg=0.0, spread_s=DEFAULT_SPREAD_S, seed=0):
    """Irregular wave field as a list of component dicts {U, T, heading_deg, phase_deg}
    (drop-in for ``DisturbanceField(waves=...)``). Components are sampled from a JONSWAP
    spectrum using **equal-energy bins with a random frequency per bin** (so the summed
    field has no artificial repeat period), uniform random phases, and cos^(2s)
    directional spreading about ``heading_deg``. U_i = omega_i * a_i is the deep-water
    surface orbital-velocity amplitude (the model's cos-horizontal / sin-vertical sum is
    the correct circular orbit). Reproducible via ``seed``. Validated by
    underwater-robotics-advisor (Fossen Ch.8 / DNV-RP-C205).

    Note: the field keeps deep-water dispersion k = omega^2/g; for Tp >= ~5 s the swell
    is only intermediate-water at the ~3 m site, so penetration is an approximation.
    """
    rng = np.random.default_rng(seed)
    wp = 2 * np.pi / Tp
    w_lo, w_hi = 0.3 * wp, 3.5 * wp

    def S(w):                                       # JONSWAP shape (unnormalized)
        sig = np.where(w <= wp, 0.07, 0.09)
        r = np.exp(-(w - wp) ** 2 / (2 * sig ** 2 * wp ** 2))
        return w ** -5.0 * np.exp(-1.25 * (wp / w) ** 4) * gamma ** r

    # equal-energy bin edges from the cumulative spectrum on a fine grid
    wg = np.linspace(w_lo, w_hi, 4000)
    Sg = S(wg)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (Sg[1:] + Sg[:-1]) * np.diff(wg))])
    cum /= cum[-1]
    edges = np.interp(np.linspace(0, 1, n + 1), cum, wg)
    w_i = edges[:-1] + rng.random(n) * np.diff(edges)     # one random freq per bin
    a_i = (Hs / 4.0) * np.sqrt(2.0 / n)                   # equal-energy, Hs-normalized
    U_i = w_i * a_i                                       # surface orbital speed amp
    phase = rng.uniform(0.0, 2 * np.pi, n)
    # cos^(2s) directional spreading about heading_deg (rejection sampling)
    th = np.empty(n)
    for i in range(n):
        while True:
            c = rng.uniform(-np.pi, np.pi)
            if rng.random() <= np.cos(c / 2.0) ** (2 * spread_s):
                th[i] = np.radians(heading_deg) + c
                break
    return [dict(U=float(U_i[i]), T=float(2 * np.pi / w_i[i]),
                 heading_deg=float(np.degrees(th[i])),
                 phase_deg=float(np.degrees(phase[i]))) for i in range(n)]


class DisturbanceField:
    """3-layer FLU disturbance field queried by hydro each substep."""

    def __init__(self, current=DEFAULT_CURRENT, waves=DEFAULT_WAVES,
                 kicks=DEFAULT_KICKS, z_surface=DEFAULT_Z_SURFACE,
                 horizon=600.0, seed=0):
        self.current = np.asarray(current, float)
        self.z_surface = float(z_surface)
        self.kicks = dict(kicks)
        self.seed = seed
        self.horizon = horizon
        # precompute wave components: (U, omega, k, dir3, phase)
        self.waves = []
        for w in waves:
            omega = 2 * np.pi / w["T"]
            k = omega * omega / G                 # deep-water dispersion
            self.waves.append((w["U"], omega, k,
                               _heading3(w["heading_deg"]),
                               np.radians(w["phase_deg"])))
        self._zhat = np.array([0.0, 0.0, 1.0])
        # master + per-layer enables
        self.enabled = True
        self.use_current = True
        self.use_waves = True
        self.use_kicks = True
        self._gen_kicks(seed)

    # ---- kicks: Poisson process precomputed over the horizon ----------------
    def _gen_kicks(self, seed):
        rng = np.random.default_rng(seed)
        rate = self.kicks["rate"]
        t, starts, forces = 0.0, [], []
        while rate > 0:
            t += rng.exponential(1.0 / rate)
            if t >= self.horizon:
                break
            d = rng.normal(size=3)
            d[2] *= 0.4                              # mostly horizontal bumps
            d /= np.linalg.norm(d) + 1e-12
            mag = rng.uniform(self.kicks["fmin"], self.kicks["fmax"])
            starts.append(t)
            forces.append(mag * d)
        self._kick_starts = np.array(starts)
        self._kick_ends = self._kick_starts + self.kicks["duration"]
        self._kick_forces = np.array(forces).reshape(-1, 3)

    # ---- queried by hydro ---------------------------------------------------
    def current_velocity(self):
        """FLU world velocity of the uniform current (0 if off)."""
        if self.enabled and self.use_current:
            return self.current.copy()
        return np.zeros(3)

    def wave_velocity(self, t, pos):
        """FLU world velocity from the waves at time t, position pos (0 if off)."""
        v = np.zeros(3)
        if not (self.enabled and self.use_waves) or not len(self.waves):
            return v
        d = max(0.0, self.z_surface - float(pos[2]))
        for U, omega, k, dir3, phase in self.waves:
            decay = np.exp(-k * d)
            ph = omega * t + phase
            v = v + U * decay * (dir3 * np.cos(ph) + self._zhat * np.sin(ph))
        return v

    def water_velocity(self, t, pos):
        """Total FLU world water velocity (current + waves)."""
        return self.current_velocity() + self.wave_velocity(t, pos)

    def external_wrench(self, t, pos):
        """Impulsive kick (world-frame force, no torque) active at time t."""
        F = np.zeros(3)
        if not (self.enabled and self.use_kicks) or self._kick_starts.size == 0:
            return F, np.zeros(3)
        i = int(np.searchsorted(self._kick_starts, t, side="right")) - 1
        if i >= 0 and t < self._kick_ends[i]:
            F = self._kick_forces[i]
        return F, np.zeros(3)

    # ---- convenience --------------------------------------------------------
    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def wave_speed_at_depth(self, depth):
        """Peak horizontal wave orbital speed at a given depth (for checks)."""
        return float(sum(U * np.exp(-k * depth) for U, _, k, _, _ in self.waves))

    def summary(self):
        cs = np.linalg.norm(self.current)
        wt = ", ".join(f"U{U:.2f}/T{2*np.pi/om:.1f}s" for U, om, *_ in self.waves)
        return (f"DisturbanceField (FLU, z_surface={self.z_surface} m):\n"
                f"  current = {self.current.tolist()} m/s (|.|={cs:.2f})\n"
                f"  waves   = [{wt}]  ({len(self.waves)} comps, deep-water decay)\n"
                f"  kicks   = rate {self.kicks['rate']}/s, "
                f"{self.kicks['fmin']:.0f}-{self.kicks['fmax']:.0f} N, "
                f"{self.kicks['duration']:.2f} s, {self._kick_starts.size} events/horizon")

    # ---- build from a sampled config ----------------------------------------
    @classmethod
    def from_config(cls, cfg, z_surface=DEFAULT_Z_SURFACE, horizon=600.0):
        return cls(current=cfg["current"], waves=cfg["waves"], kicks=cfg["kicks"],
                   z_surface=z_surface, horizon=horizon, seed=cfg["seed"])


# ---- domain randomization ----------------------------------------------------
def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def sample_config(seed):
    """Sample a bounded disturbance+model config within DR_RANGES."""
    rng = np.random.default_rng(seed)
    r = DR_RANGES
    spd = _u(rng, *r["current_speed"]);  ang = _u(rng, 0, 2 * np.pi)
    current = [spd * np.cos(ang), spd * np.sin(ang), _u(rng, *r["current_vertical"])]
    n = int(rng.integers(r["n_waves"][0], r["n_waves"][1] + 1))
    waves = [dict(U=_u(rng, *r["wave_U"]), T=_u(rng, *r["wave_T"]),
                  heading_deg=_u(rng, 0, 360), phase_deg=_u(rng, 0, 360))
             for _ in range(n)]
    fmin = _u(rng, *r["kick_fmin"]); fmax = _u(rng, *r["kick_fmax"])
    kicks = dict(rate=_u(rng, *r["kick_rate"]), fmin=fmin, fmax=max(fmax, fmin + 1),
                 duration=_u(rng, *r["kick_duration"]))
    model = dict(drag_scale=_u(rng, *r["drag_scale"]),
                 thruster_scale=_u(rng, *r["thruster_scale"]),
                 buoyancy_trim=_u(rng, *r["buoyancy_trim"]))
    return dict(current=current, waves=waves, kicks=kicks, model=model, seed=int(seed))


def randomize(seed, z_surface=DEFAULT_Z_SURFACE, horizon=600.0):
    """Return (DisturbanceField, model_params) for a fresh episode."""
    cfg = sample_config(seed)
    return DisturbanceField.from_config(cfg, z_surface=z_surface, horizon=horizon), cfg["model"]


def apply_model_params(model_params, hydro=None, base_DL=None, base_DNL=None,
                       base_buoyancy=None):
    """Apply sampled drag_scale / buoyancy_trim to a Hydrodynamics instance.

    thruster_scale is returned for the caller to apply at command time (it scales
    the thrust forces written to data.ctrl). Pass the *base* (unscaled) D_L/D_NL/
    buoyancy so repeated randomize() calls don't compound.
    """
    if hydro is not None:
        if base_DL is not None:
            hydro.D_L = np.asarray(base_DL) * model_params["drag_scale"]
        if base_DNL is not None:
            hydro.D_NL = np.asarray(base_DNL) * model_params["drag_scale"]
        if base_buoyancy is not None:
            hydro.buoyancy = base_buoyancy + model_params["buoyancy_trim"]
    return model_params["thruster_scale"]
