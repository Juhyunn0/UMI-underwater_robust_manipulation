"""Matplotlib figures replicating the paper's result plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import fossen

_COLORS = {"pid": "tab:green", "mpc": "tab:orange", "dobmpc": "tab:blue"}
_LABEL = {"pid": "PID", "mpc": "MPC", "dobmpc": "DOBMPC"}


def plot_errors(logs, path, title=""):
    fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    names = ["x error [m]", "y error [m]", "z error [m]", "yaw error [rad]"]
    idx = [0, 1, 2, 5]
    for name, lg in logs.items():
        e = lg["x"][:, :6] - lg["eta_ref"]
        e[:, 5] = fossen.wrap_angle(e[:, 5])
        for ax, i, lab in zip(axs, idx, names):
            ax.plot(lg["t"], e[:, i], color=_COLORS[name],
                    label=_LABEL[name], lw=1.2)
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
    axs[0].legend(ncol=3, loc="upper right")
    axs[0].set_title(title)
    axs[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_disturbance(log, path, title="EAOB disturbance estimation"):
    fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    names = [r"$F_x$ [N]", r"$F_y$ [N]", r"$F_z$ [N]", r"$M_z$ [Nm]"]
    idx = [0, 1, 2, 5]
    for ax, i, lab in zip(axs, idx, names):
        ax.plot(log["t"], log["w_app"][:, i], "k--", lw=1.2, label="applied")
        ax.plot(log["t"], log["w_est"][:, i], color="tab:blue", lw=1.2,
                label="EAOB estimate")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axs[0].legend(ncol=2, loc="upper right")
    axs[0].set_title(title + " (inertial frame)")
    axs[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_controls(logs, path, title="Control inputs"):
    fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    names = [r"$u_1=X_u$ [N]", r"$u_2=Y_u$ [N]", r"$u_3=Z_u$ [N]",
             r"$u_4=N_u$ [Nm]"]
    for name, lg in logs.items():
        for ax, i, lab in zip(axs, range(4), names):
            ax.plot(lg["t"], lg["u"][:, i], color=_COLORS[name],
                    label=_LABEL[name], lw=1.0)
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
    axs[0].legend(ncol=3, loc="upper right")
    axs[0].set_title(title)
    axs[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_xy(logs, path, title="Trajectory (top view, NED)"):
    fig, ax = plt.subplots(figsize=(7, 7))
    lg0 = next(iter(logs.values()))
    ax.plot(lg0["eta_ref"][:, 1], lg0["eta_ref"][:, 0], "k--", lw=1.5,
            label="reference")
    for name, lg in logs.items():
        ax.plot(lg["x"][:, 1], lg["x"][:, 0], color=_COLORS[name],
                label=_LABEL[name], lw=1.2)
    ax.set_xlabel("y east [m]")
    ax.set_ylabel("x north [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
