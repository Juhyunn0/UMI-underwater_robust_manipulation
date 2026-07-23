#!/usr/bin/env python3
"""Presentation figures: the per-run wave heading beta_bar is the MEAN of a
directional distribution, not a single wave direction.

Renders the REAL disturbance wave model (disturbance/waves.py DirectionalWaveField,
base.yaml parameters) — not a cartoon — as slide-ready PNGs:

  wave_beta_spreading.png    sea-surface snapshot + the cos^{2s} directional lobe
                             with the 21 actual direction components (main slide)
  wave_spreading_s_compare.png   same sea at s=30 (this project, swell) vs s=2
                                 (wind sea) — same total energy, only the spread
                                 differs (optional supporting slide)

Run (robust env, from bluerov2_mujoco_marinegym/):
    python tools/plot_wave_spreading.py [--beta-deg 25] [--seed 0] [--out-dir ../assets/screenshots]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # package root
from disturbance.waves import DirectionalWaveField, solve_wavenumber

# --- config/base.yaml values (waves + site blocks) ---
HS, TP, GAMMA, S_PROJ = 0.75, 12.0, 5.0, 30.0
H_DEPTH, Z_ROV = 4.0, -3.0
N_OMEGA, N_BETA = 60, 21
OMEGA_MIN, OMEGA_MAX = 0.2, 1.6

INK, INK2, ACCENT = "#16293B", "#546B80", "#E07B39"
SEA_CMAP = LinearSegmentedColormap.from_list("sea", [
    "#0F3654", "#1B567E", "#3579A2", "#6EA9C8", "#B7D9E9", "#EAF5FA"])

PATCH_X, PATCH_Y, NPX = 320.0, 200.0, 2.0        # patch size [m], grid step [m/px]


def make_field(s, beta_bar_rad, seed):
    return DirectionalWaveField(Hs=HS, Tp=TP, gamma=GAMMA, h=H_DEPTH, z_ROV=Z_ROV,
                                N_omega=N_OMEGA, N_beta=N_BETA,
                                omega_min=OMEGA_MIN, omega_max=OMEGA_MAX,
                                beta_bar=beta_bar_rad, s=s, seed=seed)


def elevation_grid(field, t=0.0):
    """eta(x, y) over the patch, vectorised over the field's internal component
    arrays (same math as DirectionalWaveField.elevation, evaluated row-wise)."""
    x = np.arange(0.0, PATCH_X, NPX)
    y = np.arange(0.0, PATCH_Y, NPX)
    eta = np.empty((y.size, x.size))
    kx = field.k_m * field.ex_m
    ky = field.k_m * field.ey_m
    for j, yj in enumerate(y):                      # rows: keep memory ~ Nx * M
        theta = np.outer(x, kx) + yj * ky - field.omega_m * t + field.eps_m
        eta[j] = np.cos(theta) @ field.a_m
    return x, y, eta


def draw_sea(ax, field, beta_deg, title, show_arrow=True):
    x, y, eta = elevation_grid(field)
    sig = HS / 4.0
    ax.imshow(eta, origin="lower", extent=[0, PATCH_X, 0, PATCH_Y],
              cmap=SEA_CMAP, vmin=-2.6 * sig, vmax=2.6 * sig,
              interpolation="bilinear", rasterized=True)
    if show_arrow:
        th = np.deg2rad(beta_deg)
        cx, cy, L = PATCH_X / 2, PATCH_Y / 2, 55.0
        ax.annotate("", xytext=(cx - 0.55 * L * np.cos(th), cy - 0.55 * L * np.sin(th)),
                    xy=(cx + L * np.cos(th), cy + L * np.sin(th)),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.9",
                                    lw=4.5, color=ACCENT, shrinkA=0, shrinkB=0))
        ax.text(cx + 1.28 * L * np.cos(th), cy + 1.28 * L * np.sin(th),
                r"$\bar\beta$", color="#FFDCC2", fontsize=22,
                ha="center", va="center", fontweight="bold")
    # 50 m scale bar
    ax.plot([PATCH_X - 62, PATCH_X - 12], [12, 12], color="#EAF5FA", lw=3,
            solid_capstyle="butt")
    ax.text(PATCH_X - 37, 19, "50 m", color="#EAF5FA", fontsize=10, ha="center")
    ax.set_title(title, fontsize=13.5, color=INK, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#D9E3EA")


def draw_lobe(ax, field, beta_deg, s):
    """Polar D(beta) lobe + the sim's 21 discrete direction components."""
    bb = np.deg2rad(beta_deg)
    fine = np.linspace(-np.pi / 2, np.pi / 2, 361)
    D = np.cos(fine / 2.0) ** (2.0 * s)
    ax.plot(bb + fine, D, color="#3579A2", lw=2.2, zorder=3)
    ax.fill(np.concatenate([[bb], bb + fine, [bb]]),
            np.concatenate([[0], D, [0]]), color="#3579A2", alpha=0.25, zorder=2)
    # the 21 actual grid components (18 deg apart); weight = normalised D
    Dg = field.D / field.D.max()
    for b, w in zip(field.beta_grid, Dg):
        if w > 1e-4:
            ax.plot([b, b], [0, w], color=INK, lw=1.6, alpha=0.6, zorder=4)
            ax.plot(b, w, "o", ms=6.5, color=INK, mec="white", mew=1.2, zorder=5)
        else:
            ax.plot(b, 1.30, "o", ms=3.2, color="#C4D2DC", zorder=1, clip_on=False)
    # mean heading + half-power width
    ax.annotate("", xytext=(bb, 0), xy=(bb, 1.13),
                arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
                                lw=3, color=ACCENT, shrinkA=0, shrinkB=0))
    ax.text(bb, 1.32, r"$\bar\beta$", color=ACCENT, fontsize=20,
            ha="center", va="center", fontweight="bold")
    hw = 2.0 * np.arccos(0.5 ** (1.0 / (2.0 * s)))
    arc = np.linspace(bb - hw, bb + hw, 60)
    ax.plot(arc, np.full_like(arc, 0.5), color=ACCENT, lw=2.4, zorder=6)
    ax.text(bb + hw * 2.6, 0.55, f"half-power\n$\\pm${np.degrees(hw):.0f}°",
            color="#9C4E1B", fontsize=10.5, ha="center", va="center")
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_rlim(0, 1.38); ax.set_rticks([])
    ax.set_thetagrids([0, 90, 180, 270], ["+x (0°)", "+y", "180°", ""],
                      fontsize=10, color=INK2)
    ax.grid(color="#E7EEF3", lw=0.8)
    ax.spines["polar"].set_color("#D9E3EA")
    ax.set_title("directional energy  $D(\\beta)\\propto\\cos^{2s}"
                 "\\frac{\\beta-\\bar\\beta}{2}$,  $s=%g$" % s,
                 fontsize=13.5, color=INK, pad=18)


def fig_vectors(beta_deg, seed, out):
    """One arrow + one number per direction component — no rendered sea, just the
    sim's actual 21 direction weights (field.D * dBeta, sums to 1)."""
    bb = np.deg2rad(beta_deg)
    field = make_field(S_PROJ, bb, seed)
    shares = field.D * field.dBeta                    # energy share per component
    rel_deg = np.round(np.degrees(field.beta_grid - bb)).astype(int)   # -180..180

    fig = plt.figure(figsize=(12.8, 5.4))
    axf = fig.add_axes([0.02, 0.02, 0.44, 0.82]); axf.set_aspect("equal")
    axb = fig.add_axes([0.54, 0.14, 0.43, 0.68])

    # ---- left: vector fan, arrow length = energy share (real numbers) ----
    smax = shares.max()
    circ = np.linspace(0, 2 * np.pi, 181)
    axf.plot(1.06 * np.cos(circ), 1.06 * np.sin(circ), color="#E7EEF3", lw=1)
    for ang, sh, rd in zip(field.beta_grid, shares, rel_deg):
        ux, uy = np.cos(ang), np.sin(ang)
        if sh >= 5e-4:                                # visible arrow + its number
            L = sh / smax
            main = (rd == 0)
            head = ("-|>,head_width=0.32,head_length=0.62" if main
                    else "-|>,head_width=0.16,head_length=0.34")
            axf.annotate("", xytext=(0, 0), xy=(L * ux, L * uy), zorder=5,
                         arrowprops=dict(arrowstyle=head,
                                         lw=4.5 if main else 2.4,
                                         color=ACCENT if main else INK,
                                         shrinkA=0, shrinkB=0))
            # labels on a fixed outer radius so they never collide with the arrows
            lab = (r"$\bar\beta$" + f"\n{sh*100:.1f}%") if main else f"{rd:+d}°\n{sh*100:.1f}%"
            r_lab = 1.36 if main else 1.27
            axf.plot([L * ux, 1.10 * ux], [L * uy, 1.10 * uy],
                     color="#C4D2DC", lw=0.9, ls=":", zorder=1)
            axf.text(r_lab * ux, r_lab * uy, lab, fontsize=13.5 if main else 11,
                     color=ACCENT if main else INK, ha="center", va="center",
                     fontweight="bold" if main else "normal", linespacing=1.25)
        else:                                         # ~zero components: gray ticks
            axf.plot([1.03 * ux, 1.09 * ux], [1.03 * uy, 1.09 * uy],
                     color="#C4D2DC", lw=1.6)
    axf.text(0, -1.30, "remaining 14 directions: < 0.05% each  (≈ 0)",
             fontsize=10.5, color=INK2, ha="center")
    axf.text(0, -1.44, "arrow length = energy share (actual sim weights)",
             fontsize=10.5, color=INK2, ha="center")
    axf.set_xlim(-1.62, 1.62); axf.set_ylim(-1.55, 1.52); axf.axis("off")

    # ---- right: all 21 numbers as labelled bars ----
    order = np.argsort(rel_deg)
    xs, hs = rel_deg[order], shares[order] * 100.0
    colors = [ACCENT if d == 0 else INK for d in xs]
    axb.bar(xs, hs, width=12.5, color=colors, edgecolor="white", lw=0.8)
    for d, h in zip(xs, hs):
        if h >= 0.05:
            axb.text(d, h + 1.2, f"{h:.1f}", fontsize=11, color=INK,
                     ha="center", fontweight="bold" if d == 0 else "normal")
    within = shares[np.abs(rel_deg) <= 18].sum() * 100.0
    axb.annotate(f"{within:.0f}% of the energy\nwithin " + r"$\bar\beta\pm$18°",
                 xy=(0, 42), xytext=(78, 43), fontsize=12.5, color=INK,
                 va="center", ha="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    axb.set_xticks(np.arange(-180, 181, 36))
    axb.set_xlabel(r"component direction relative to $\bar\beta$  [deg]", fontsize=11.5)
    axb.set_ylabel("energy share  [%]", fontsize=11.5)
    axb.set_ylim(0, 56); axb.set_xlim(-192, 192)
    axb.tick_params(colors=INK2, labelsize=10)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axb.spines[sp].set_color("#D9E3EA")
    axb.set_title("all 21 direction components of the sim's wave field  (18° grid)",
                  fontsize=12.5, color=INK, pad=10)

    fig.text(0.02, 0.93, r"$\bar\beta$ is the MEAN of the direction components — "
             "not the only wave direction", fontsize=17.5, color=INK, fontweight="bold")
    fig.text(0.02, 0.875, r"$D(\beta)\propto\cos^{2s}\frac{\beta-\bar\beta}{2}$,"
             f"  $s={S_PROJ:g}$   (disturbance/waves.py, config/base.yaml)",
             fontsize=12, color=INK2)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    # the same numbers on stdout, for copy-paste
    print(f"[plot_wave_spreading] wrote {out}")
    print("    dir rel to beta_bar : energy share")
    for d, h in zip(xs, hs):
        print(f"    {d:+4d} deg : {h:7.3f} %")


def fig_main(beta_deg, seed, out):
    field = make_field(S_PROJ, np.deg2rad(beta_deg), seed)
    fig = plt.figure(figsize=(12.6, 4.9))
    ax1 = fig.add_axes([0.035, 0.06, 0.52, 0.78])
    ax2 = fig.add_axes([0.615, 0.10, 0.33, 0.70], projection="polar")
    draw_sea(ax1, field, beta_deg,
             "sea-surface snapshot — crests travel NEAR $\\bar\\beta$, not exactly along it")
    draw_lobe(ax2, field, beta_deg, S_PROJ)
    fig.text(0.035, 0.955, r"$\bar\beta$ is the MEAN wave heading of a spread — "
             "not a single wave direction", fontsize=17, color=INK, fontweight="bold")
    fig.text(0.615, 0.045, "dots = the sim's 21 direction components (18° grid)",
             fontsize=10.5, color=INK2)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_wave_spreading] wrote {out}")


def fig_compare(beta_deg, seed, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, s, lab in zip(axes, (S_PROJ, 2.0),
                          (f"s = {S_PROJ:g} — swell (this project)",
                           "s = 2 — wind sea (same total energy)")):
        draw_sea(ax, make_field(s, np.deg2rad(beta_deg), seed), beta_deg, lab)
    fig.suptitle("narrow vs wide directional spreading — only $s$ differs",
                 fontsize=16, color=INK, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_wave_spreading] wrote {out}")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-deg", type=float, default=25.0, help="mean wave heading")
    ap.add_argument("--seed", type=int, default=0, help="component phase seed")
    ap.add_argument("--out-dir", default=os.path.join(here, "..", "assets", "screenshots"))
    ap.add_argument("--sea-lobe", action="store_true",
                    help="also render the rendered-sea + polar-lobe variant")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fig_vectors(args.beta_deg, args.seed,
                os.path.join(args.out_dir, "wave_beta_vectors.png"))
    fig_compare(args.beta_deg, args.seed,
                os.path.join(args.out_dir, "wave_spreading_s_compare.png"))
    if args.sea_lobe:
        fig_main(args.beta_deg, args.seed,
                 os.path.join(args.out_dir, "wave_beta_spreading.png"))


if __name__ == "__main__":
    main()
