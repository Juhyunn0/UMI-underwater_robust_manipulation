#!/usr/bin/env python3
"""state_noise.py — deliberate Gaussian noise on the state the LOW controller
is handed (--state-noise, 2026-10-19; HARDWARE-UNVERIFIED).

The question it answers: how does a follower (pid / mpc_tuned / rl / rl_pwm /
...) behave when its state measurement is worse than the station's? Every LOW
reads the vehicle state through ONE call — ``ctrl.step(eta, nu, nudot, t)`` —
so the noise is added there and nowhere else (workers.MpcWorker.tick, the
``meas_ctrl`` seam the imu_dr / EKF switches already use):

    * the controller, the path cursor and the coast retarget see the noisy
      state (they have to agree — the cursor projects the same vehicle the
      controller flies);
    * the plot, the CSV px/py/..., the DP policy's proprio (unless
      ``policy``, below) and every interlock (runtime fault, per-thruster,
      attitude, replay/policy divergence) stay on the un-noised state. The
      ONE exception is the path cursor's own projection leash / completion
      and CSV e_along / e_cross, which judge the noised pose the controller
      flies; the true gap is then bounded by leash + |noise| (keep pos_m well
      under the leash).

``policy`` (set ONLY by --state-noise-policy — the station refuses it as a
yaml key; 2026-10-19; HARDWARE-UNVERIFIED) is the one opt-in widening: the DP policy then sees the SAME draw the controller
was handed on that tick — on the eta MpcWorker._emit_policy_state sends its
worker (the proprio rows) and keeps (the anchor history a plan is composed
on), and on the policy START pose (the wrt_start rows). One noisy estimator,
never a second draw: a tick that adds nothing to the controller adds nothing
to the DP. The interlocks, the plot and CSV px/py stay un-noised;
plans.jsonl ``sn_policy`` records what each plan's rows and anchor carried.

It is STATE-level noise, not sensor-level: position noise is not
differentiated into the velocity (that channel has its own sigma), and the
angular acceleration in nudot (the assembler's finite difference of the gyro)
is left alone. Each channel is a first-order Gauss-Markov process with
stationary standard deviation sigma; ``tau_s: 0`` = white per control tick.

Frames are the controller's: eta = (x, y, z) NED world [m], (roll, pitch, yaw)
[rad]; nu = (u, v, w) FRD body [m/s], (p, q, r) [rad/s]; nudot[0:3] FRD
[m/s^2]. The config is in degrees for the angles.

Every default sigma below is a CHOICE ("a little noise"), not a measurement.
For scale: the measured tag position noise is 0.8 mm (config/hw_mpc.yaml,
vel_lp_alpha note), so pos_m 0.005 is ~6x that [유도].
"""

from __future__ import annotations

import math

import numpy as np

#: hw_mpc.yaml `state_noise:` keys. `scale` multiplies every sigma (the
#: --state-noise SCALE argument overrides it). All sigmas are 1-sigma.
DEFAULTS = {
    "scale": 1.0,
    "pos_m": 0.005,        # x, y            [예측: choice, ~6x tag noise]
    "z_m": 0.005,          # z               [예측: choice]
    "rp_deg": 0.5,         # roll, pitch     [예측: choice]
    "yaw_deg": 1.0,        # yaw             [예측: choice]
    "vel_m_s": 0.01,       # u, v, w         [예측: choice]
    "rate_deg_s": 0.5,     # p, q, r         [예측: choice]
    "acc_m_s2": 0.0,       # nudot u, v, w (only the EAOB reads it)
    "tau_s": 0.0,          # 0 = white per tick; > 0 = Gauss-Markov correlation time
    "seed": None,          # None = a fresh seed per engagement (recorded in meta)
    "policy": False,       # True = the DP sees the SAME draw; set ONLY by --state-noise-policy
}

#: HARD ceilings on the EFFECTIVE sigma (scale x sigma). Over one = the config
#: is refused at load / at --state-noise. This mode is "a little noise"; a
#: sigma that could walk the vehicle into a wall is a different experiment
#: and needs its own review. [예측: engineering bound, not a measurement]
LIMITS = {
    "pos_m": 0.05,
    "z_m": 0.05,
    "rp_deg": 5.0,
    "yaw_deg": 5.0,
    "vel_m_s": 0.10,
    "rate_deg_s": 10.0,
    "acc_m_s2": 1.0,
}

#: correlation time ceiling: a slower process is a bias, not noise
TAU_MAX_S = 60.0

SIGMA_KEYS = ("pos_m", "z_m", "rp_deg", "yaw_deg", "vel_m_s", "rate_deg_s",
              "acc_m_s2")


def resolve(raw: dict | None, scale: float | None = None) -> dict:
    """hw_mpc.yaml `state_noise:` (+ an optional --state-noise SCALE) -> a full
    config dict. Unknown keys, negative / non-finite values and an effective
    sigma over LIMITS RAISE (the dvl rule: a typo must not fly a default)."""
    raw = dict(raw or {})
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown state_noise keys {sorted(unknown)}; "
                         f"known: {sorted(DEFAULTS)}")
    cfg = dict(DEFAULTS)
    cfg.update(raw)
    if scale is not None:
        cfg["scale"] = scale
    for k in ("scale", "tau_s") + SIGMA_KEYS:
        try:
            v = float(cfg[k])
        except (TypeError, ValueError):
            raise ValueError(f"state_noise.{k} must be a number, got "
                             f"{cfg[k]!r}") from None
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"state_noise.{k} must be finite and >= 0, "
                             f"got {v}")
        cfg[k] = v
    if not isinstance(cfg["policy"], bool):
        raise ValueError(f"state_noise.policy must be true or false, got "
                         f"{cfg['policy']!r}")
    if cfg["tau_s"] > TAU_MAX_S:
        raise ValueError(f"state_noise.tau_s {cfg['tau_s']} > {TAU_MAX_S} s "
                         f"(that is a bias, not noise)")
    for k in SIGMA_KEYS:
        eff = cfg["scale"] * cfg[k]
        if eff > LIMITS[k]:
            raise ValueError(
                f"state_noise: effective {k} = scale {cfg['scale']:g} x "
                f"{cfg[k]:g} = {eff:g} > the {LIMITS[k]:g} ceiling "
                f"(state_noise.LIMITS)")
    if cfg["seed"] is not None:
        try:
            cfg["seed"] = int(cfg["seed"])
        except (TypeError, ValueError):
            raise ValueError(f"state_noise.seed must be an integer or null, "
                             f"got {cfg['seed']!r}") from None
        if cfg["seed"] < 0:
            raise ValueError("state_noise.seed must be >= 0")
    return cfg


class StateNoise:
    """15 independent Gauss-Markov channels: eta (6), nu (6), nudot[0:3] (3).

    ``reset(seed)`` at every engagement (a fresh draw unless the config pins
    a seed); ``apply(meas)`` once per controller tick returns a NEW meas dict
    and the 15-vector it added. The input dict and its arrays are never
    mutated — the caller's `meas` is the truth every interlock reads."""

    N = 15

    def __init__(self, cfg: dict, dt: float):
        self.cfg = resolve(cfg)
        self.dt = float(dt)
        if not (self.dt > 0.0):
            raise ValueError(f"StateNoise: dt must be > 0, got {dt}")
        c, s = self.cfg, self.cfg["scale"]
        d2r = math.pi / 180.0
        self.sigma = s * np.array(
            [c["pos_m"], c["pos_m"], c["z_m"],
             c["rp_deg"] * d2r, c["rp_deg"] * d2r, c["yaw_deg"] * d2r,
             c["vel_m_s"], c["vel_m_s"], c["vel_m_s"],
             c["rate_deg_s"] * d2r, c["rate_deg_s"] * d2r,
             c["rate_deg_s"] * d2r,
             c["acc_m_s2"], c["acc_m_s2"], c["acc_m_s2"]], float)
        tau = c["tau_s"]
        self.a = math.exp(-self.dt / tau) if tau > 0.0 else 0.0
        self.b = math.sqrt(max(0.0, 1.0 - self.a * self.a))
        self.seed = None
        self.n_applied = 0
        self.reset(c["seed"])

    def reset(self, seed: int | None = None) -> int:
        """Cold state (a stationary draw, so the first tick is not special)
        and a new generator. -> the seed used (record it: it reproduces the
        sequence)."""
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2 ** 63))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._x = self.sigma * self._rng.standard_normal(self.N)
        self.n_applied = 0
        return self.seed

    @property
    def zero(self) -> bool:
        """True when every sigma is 0 (the mode would add nothing)."""
        return not bool(np.any(self.sigma > 0.0))

    def _next(self) -> np.ndarray:
        if self.n_applied > 0:
            self._x = (self.a * self._x
                       + self.b * self.sigma * self._rng.standard_normal(self.N))
        self.n_applied += 1
        return self._x.copy()

    def apply(self, meas: dict) -> tuple[dict, np.ndarray]:
        """-> (noisy copy of meas, the 15-vector added). ``meas`` needs eta
        (6), nu (6) and nudot (6); other keys are carried over unchanged."""
        d = self._next()
        out = dict(meas)
        out["eta"] = np.asarray(meas["eta"], float) + d[0:6]
        out["nu"] = np.asarray(meas["nu"], float) + d[6:12]
        nd = np.array(meas["nudot"], float, copy=True)
        nd[0:3] += d[12:15]
        out["nudot"] = nd
        return out, d

    def describe(self) -> str:
        c, s = self.cfg, self.cfg["scale"]
        corr = (f"tau {c['tau_s']:g} s" if c["tau_s"] > 0 else "white")
        return (f"scale {s:g}: xy {s * c['pos_m'] * 1e3:.1f} mm, "
                f"z {s * c['z_m'] * 1e3:.1f} mm, rp {s * c['rp_deg']:.2f} deg, "
                f"yaw {s * c['yaw_deg']:.2f} deg, uvw "
                f"{s * c['vel_m_s'] * 1e3:.1f} mm/s, pqr "
                f"{s * c['rate_deg_s']:.2f} deg/s, acc "
                f"{s * c['acc_m_s2']:.3f} m/s^2 (1-sigma, {corr})"
                + (" + DP proprio/anchor (same draw)" if c.get("policy")
                   else ""))

    def meta(self) -> dict:
        return {"config": dict(self.cfg),
                "sigma_effective": {
                    "frame": "controller: eta NED world [m, rad], nu FRD "
                             "body [m/s, rad/s], nudot[0:3] FRD [m/s^2]",
                    "values": [float(v) for v in self.sigma]},
                "gm_a": self.a, "dt": self.dt,
                "seed": self.seed, "n_applied": int(self.n_applied),
                "describe": self.describe()}
