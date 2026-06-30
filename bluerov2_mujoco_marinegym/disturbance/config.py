#!/usr/bin/env python3
"""YAML config loader for the disturbance environment + comparison experiment.

`load_config(path)` parses a YAML (e.g. config/base.yaml) into a `Config` with:
  * `.dist`       — DistConfig consumed by DisturbanceEnv (physics params, derived
                    radians / sigma_inf folded in)
  * `.sim`        — dt, T_sim, log_hz
  * `.experiment` — run matrix (primary/secondary blocks, controllers, seeds, ...)

Config-driven so modes / parameters change with no code edit. Validates the
geometry (-h <= z_ROV <= 0) so the operating depth sits inside the water column.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import yaml


@dataclass
class DistConfig:
    # waves
    Hs: float
    Tp: float
    gamma: float
    h: float
    z_ROV: float
    N_omega: int
    N_beta: int
    omega_min: float
    omega_max: float
    beta_bar: float            # radians (derived from beta_bar_deg)
    s: float
    # current
    V_bar_c: float
    theta_c: float             # radians (derived from theta_c_deg)
    tau: float
    sigma_inf: float           # derived = sigma_inf_frac * V_bar_c
    v_z_range: Tuple[float, float]
    # inertia / Froude-Krylov
    rho: float
    vol: float
    C_M: float
    fk_mode: str
    added_mass_xyz: Tuple[float, float, float] = (5.5, 12.7, 14.57)  # for morison_ca only


@dataclass
class Config:
    dist: DistConfig
    sim: dict
    experiment: dict
    raw: dict = field(default_factory=dict)
    path: str = ""


def load_config(path):
    with open(path) as f:
        d = yaml.safe_load(f)

    site = d["site"]
    waves = d["waves"]
    cur = d["current"]
    inertia = d.get("inertia", {})
    sim = d["sim"]

    h = float(site["h"])
    z_ROV = float(site["z_ROV"])
    assert -h <= z_ROV <= 0.0, (
        f"z_ROV={z_ROV} must lie in [-h, 0] = [{-h}, 0] (operating depth inside the "
        f"water column); got h={h}")

    V_bar_c = float(cur["V_bar_c"])
    dist = DistConfig(
        Hs=float(waves["Hs"]), Tp=float(waves["Tp"]), gamma=float(waves["gamma"]),
        h=h, z_ROV=z_ROV,
        N_omega=int(waves["N_omega"]), N_beta=int(waves["N_beta"]),
        omega_min=float(waves["omega_min"]), omega_max=float(waves["omega_max"]),
        beta_bar=float(np.radians(waves.get("beta_bar_deg", 0.0))),
        s=float(waves["s"]),
        V_bar_c=V_bar_c,
        theta_c=float(np.radians(cur.get("theta_c_deg", 0.0))),
        tau=float(cur["tau"]),
        sigma_inf=float(cur["sigma_inf_frac"]) * V_bar_c,
        v_z_range=tuple(float(x) for x in cur["v_z_range"]),
        rho=float(inertia.get("rho", 1025.0)),
        vol=float(inertia.get("vol", 0.0113459)),
        C_M=float(inertia.get("C_M", 1.0)),
        fk_mode=str(inertia.get("fk_mode", "froude_krylov")),
        added_mass_xyz=tuple(float(x) for x in
                             inertia.get("added_mass_xyz", (5.5, 12.7, 14.57))),
    )
    assert dist.fk_mode in ("froude_krylov", "morison_ca", "off"), dist.fk_mode

    return Config(dist=dist, sim=dict(sim), experiment=dict(d["experiment"]),
                  raw=d, path=os.path.abspath(path))
