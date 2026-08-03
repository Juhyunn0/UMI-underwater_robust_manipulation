#!/usr/bin/env python3
"""YAML config loader for the disturbance environment + comparison experiment.

`load_config(path)` parses a YAML (e.g. config/base.yaml) into a `Config` with:
  * `.dist`         — DistConfig consumed by DisturbanceEnv (physics params, derived
                      radians / sigma_inf folded in)
  * `.sim`          — dt, T_sim, log_hz
  * `.experiment`   — run matrix (primary/secondary blocks, controllers, seeds, ...)
  * `.wave_configs` — LIST of DistConfig, one per wave spectrum in `waves:` (length 1
                      for the single-mapping form). `.dist` == `.wave_configs[0]`, so
                      every existing single-wave consumer is unchanged.
  * `.wave_labels`  — short filename-safe label per wave spectrum (for run_compare's
                      per-wave output subfolders and the cross-wave summary).
  * `.wave_specs`   — the raw per-wave YAML dicts (metadata / provenance).

`waves:` may be EITHER a single mapping (the original form -> one wave config) OR a
YAML list of mappings (-> N wave configs the comparison sweeps over). Only the wave
SPECTRUM fields vary between list entries; site / current / inertia are shared.

Config-driven so modes / parameters change with no code edit. Validates the
geometry (-h <= z_ROV <= 0) so the operating depth sits inside the water column.
"""
import os
import re
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
    # one DistConfig / label / raw-dict per wave spectrum (length 1 for the single
    # mapping form). wave_configs[0] is dist -> single-wave consumers are unaffected.
    wave_configs: List[DistConfig] = field(default_factory=list)
    wave_labels: List[str] = field(default_factory=list)
    wave_specs: List[dict] = field(default_factory=list)


def _sanitize(s):
    """Filename-safe token: keep [A-Za-z0-9._-], collapse the rest to '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-_") or "wave"


def _wave_label(waves):
    """Short label for one wave spectrum: explicit `label:` if given, else Hs/Tp."""
    lab = waves.get("label")
    if lab:
        return _sanitize(lab)
    return _sanitize(f"Hs{float(waves['Hs']):g}_Tp{float(waves['Tp']):g}")


def _build_dist(waves, h, z_ROV, cur, inertia, V_bar_c):
    """Assemble a DistConfig from ONE wave spectrum + the shared site/current/inertia
    blocks (so every wave in a list reuses the same current + Froude-Krylov params)."""
    return DistConfig(
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


def load_config(path):
    with open(path) as f:
        d = yaml.safe_load(f)

    site = d["site"]
    waves_raw = d["waves"]
    cur = d["current"]
    inertia = d.get("inertia", {})
    sim = d["sim"]

    # `waves:` is a single mapping (one config) or a list of mappings (N configs the
    # comparison sweeps over). Normalise to a list; only the SPECTRUM fields vary.
    if isinstance(waves_raw, dict):
        wave_specs = [waves_raw]
    elif isinstance(waves_raw, (list, tuple)):
        if not waves_raw:
            raise ValueError("`waves:` is an empty list; give at least one wave spec")
        wave_specs = []
        for i, w in enumerate(waves_raw):             # validate BEFORE dict(w) copies it
            if not isinstance(w, dict):
                raise ValueError(f"`waves:` list entry {i} must be a mapping, "
                                 f"got {type(w).__name__}")
            wave_specs.append(dict(w))
    else:
        raise ValueError("`waves:` must be a mapping or a list of mappings, "
                         f"got {type(waves_raw).__name__}")

    h = float(site["h"])
    z_ROV = float(site["z_ROV"])
    assert -h <= z_ROV <= 0.0, (
        f"z_ROV={z_ROV} must lie in [-h, 0] = [{-h}, 0] (operating depth inside the "
        f"water column); got h={h}")

    V_bar_c = float(cur["V_bar_c"])
    wave_configs = [_build_dist(w, h, z_ROV, cur, inertia, V_bar_c) for w in wave_specs]
    wave_labels = [_wave_label(w) for w in wave_specs]
    for dc in wave_configs:
        assert dc.fk_mode in ("froude_krylov", "morison_ca", "off"), dc.fk_mode

    dist = wave_configs[0]                                    # single-wave back-compat
    return Config(dist=dist, sim=dict(sim), experiment=dict(d["experiment"]),
                  raw=d, path=os.path.abspath(path),
                  wave_configs=wave_configs, wave_labels=wave_labels,
                  wave_specs=wave_specs)
