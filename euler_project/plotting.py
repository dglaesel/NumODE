"""Plotting utilities for the Euler ODE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "lines.linewidth": 2.2,
    }
)
plt.style.use("ggplot")
DEFAULT_LW = 2.2

Array = np.ndarray


FIGS_DIR = Path("figs")


def ensure_dir(path: Path | str) -> Path:
    """Ensure that the directory exists and return its path object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def savefig(fig: Figure, filepath: Path | str) -> None:
    """Save a Matplotlib figure with sensible layout.

    - Applies tight_layout only for 2D figures (3D axes can misbehave).
    - Saves both PNG (300 dpi) and PDF (vector).
    """

    p = Path(filepath)
    has_3d = any(getattr(ax, "name", "") == "3d" for ax in fig.axes)
    if not has_3d:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")


def plot_param_sweep(t: Array, trajectories: Sequence[Array], qs: Sequence[float]) -> Figure:
    """Plot trajectories from the cubic parameter sweep, with clean layout."""

    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_pairs = sorted(zip(trajectories, qs), key=lambda z: z[1])
    for X, q in sorted_pairs:
        ax.plot(t, X[:, 0], label=f"q = {q}", linewidth=DEFAULT_LW)
        # subtle equilibrium guide (no legend entry)
        ax.axhline(np.sqrt(q), color="0.6", linestyle="--", linewidth=1.0, alpha=0.4, label="_nolegend_")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Cubic ODE parameter sweep (equilibria shown)")
    ax.grid(True)
    # Figure-level legend below the plot for clarity
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        title="Parameter",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.038),
        ncol=min(3, len(handles)),
        frameon=True,
    )
    fig.add_artist(leg)
    fig.subplots_adjust(right=0.95, left=0.12, top=0.9, bottom=0.22)
    return fig


def plot_compare_methods(
    t_euler_fine: Array,
    x_euler_fine: Array,
    t_euler_coarse: Array,
    x_euler_coarse: Array,
    t_ref: Array,
    x_ref: Array,
    q: float,
) -> Figure:
    """Create a comparison figure across numerical methods for a given q."""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_ref, x_ref[:, 0], label="LSODA", linewidth=DEFAULT_LW)
    ax.plot(
        t_euler_fine,
        x_euler_fine[:, 0],
        label=r"Euler $\tau$=0.01",
        linestyle="--",
        linewidth=DEFAULT_LW,
    )
    ax.plot(
        t_euler_coarse,
        x_euler_coarse[:, 0],
        label=r"Euler $\tau$=0.1",
        linestyle=":",
        linewidth=DEFAULT_LW,
    )
    y_eq = np.sqrt(q)
    ax.axhline(y_eq, linestyle="--", linewidth=1.2, alpha=0.6, color="0.5", label=r"$\sqrt{q}$")

    ef = np.interp(t_ref, t_euler_fine, x_euler_fine[:, 0])
    ec = np.interp(t_ref, t_euler_coarse, x_euler_coarse[:, 0])
    err_f = np.abs(ef - x_ref[:, 0])
    err_c = np.abs(ec - x_ref[:, 0])

    ax2 = ax.twinx()
    # Show error as soft bands to reduce visual competition with solution curves
    ax2.fill_between(t_ref, 0.0, err_f, color="tab:red", alpha=0.18, label=r"$|err|$ $\tau$=0.01")
    ax2.fill_between(t_ref, 0.0, err_c, color="tab:blue", alpha=0.18, label=r"$|err|$ $\tau$=0.1")
    ax2.set_ylabel("abs. error (Euler vs LSODA)")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(f"Method comparison for q = {q}")
    ax.grid(True)

    # Figure-level legends below the axes: separate boxes for Solution and Error
    sol_handles, sol_labels = ax.get_legend_handles_labels()
    err_handles, err_labels = ax2.get_legend_handles_labels()
    leg1 = fig.legend(
        sol_handles,
        sol_labels,
        title="Solution",
        loc="lower left",
        bbox_to_anchor=(0.02, -0.035),
        ncol=max(1, len(sol_handles)),
        frameon=True,
    )
    leg2 = fig.legend(
        err_handles,
        err_labels,
        title="Error",
        loc="lower right",
        bbox_to_anchor=(0.98, -0.035),
        ncol=max(1, len(err_handles)),
        frameon=True,
    )
    fig.add_artist(leg1)
    fig.add_artist(leg2)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.28)
    return fig


def plot_lorenz_with_timecolor(t: Array, X: Array, t2: Array, X2: Array) -> Figure:
    """Plot Lorenz trajectories with clear contrast – no inset."""

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Baseline: thin gray line only (avoid busy scatter)
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color="0.4", linewidth=1.2, alpha=0.7, label="Baseline")

    # Perturbed: highlight
    ax.plot(X2[:, 0], X2[:, 1], X2[:, 2], color="crimson", linewidth=2.6, label=r"$x_2(0)=5.01$ perturbed")
    ax.scatter3D(X[0, 0], X[0, 1], X[0, 2], color="black", s=18, marker="o")
    ax.scatter3D(X2[0, 0], X2[0, 1], X2[0, 2], color="crimson", s=26, marker="^")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("Lorenz sensitivity to perturbing $x_2(0)$")
    ax.view_init(elev=22, azim=-45)

    # Legend below the plot (figure-level)
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(handles),
        frameon=True,
    )
    fig.add_artist(leg)
    fig.subplots_adjust(bottom=0.22)

    return fig


def plot_lorenz_separation(t: Array, X: Array, X2: Array) -> Figure:
    """Plot the separation |ΔX| between two Lorenz trajectories over time."""

    sep = np.linalg.norm(X2 - X, axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, sep, color="crimson", linewidth=DEFAULT_LW)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"|$\Delta X$|(t)")
    ax.set_title("Lorenz trajectory separation")
    ax.grid(True)
    fig.subplots_adjust(bottom=0.14, left=0.12, right=0.95, top=0.88)
    return fig


__all__ = [
    "ensure_dir",
    "savefig",
    "plot_param_sweep",
    "plot_compare_methods",
    "plot_lorenz_with_timecolor",
    "plot_lorenz_separation",
]
