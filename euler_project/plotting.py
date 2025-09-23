"""Plotting utilities for the Euler ODE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
    """Save a Matplotlib figure with tight layout."""

    p = Path(filepath)
    fig.tight_layout()
    # Always save PNG at 300 dpi
    fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    # Also save vector PDF
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")


def plot_param_sweep(t: Array, trajectories: Sequence[Array], qs: Sequence[float]) -> Figure:
    """Plot multiple trajectories from the cubic parameter sweep."""

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_pairs = sorted(zip(trajectories, qs), key=lambda z: z[1])
    for X, q in sorted_pairs:
        ax.plot(t, X[:, 0], label=f"q = {q}", linewidth=DEFAULT_LW)
        ax.axhline(np.sqrt(q), linestyle="--", linewidth=1.2, alpha=0.7, label="_nolegend_")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Cubic ODE parameter sweep (equilibria shown)")
    ax.grid(True)
    ax.legend(title="Parameter", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(right=0.82)

    iax = inset_axes(ax, width="38%", height="45%", loc="lower left", borderpad=1.1)
    for X, _ in sorted_pairs:
        iax.plot(t, X[:, 0])
    iax.set_xlim(0, 1.0)
    iax.grid(True, alpha=0.25)
    mark_inset(ax, iax, loc1=3, loc2=1, fc="none", ec="0.5", alpha=0.6)
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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_ref, x_ref[:, 0], label="LSODA", linewidth=DEFAULT_LW)
    ax.plot(
        t_euler_fine,
        x_euler_fine[:, 0],
        label="Euler τ=0.01",
        linestyle="--",
        linewidth=DEFAULT_LW,
    )
    ax.plot(
        t_euler_coarse,
        x_euler_coarse[:, 0],
        label="Euler τ=0.1",
        linestyle=":",
        linewidth=DEFAULT_LW,
    )
    y_eq = np.sqrt(q)
    ax.axhline(y_eq, linestyle="--", linewidth=1.2, alpha=0.7, label="√q")

    ef = np.interp(t_ref, t_euler_fine, x_euler_fine[:, 0])
    ec = np.interp(t_ref, t_euler_coarse, x_euler_coarse[:, 0])
    err_f = np.abs(ef - x_ref[:, 0])
    err_c = np.abs(ec - x_ref[:, 0])

    ax2 = ax.twinx()
    ax2.plot(t_ref, err_f, alpha=0.35, label="|err| τ=0.01")
    ax2.plot(t_ref, err_c, alpha=0.35, label="|err| τ=0.1")
    ax2.set_ylabel("abs. error (Euler vs LSODA)")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(f"Method comparison for q = {q}")
    ax.grid(True)
    ax.legend(title="Solution", loc="upper left")
    ax2.legend(title="Error", loc="upper right")
    return fig


def plot_lorenz_with_timecolor(t: Array, X: Array, t2: Array, X2: Array) -> Figure:
    """Plot Lorenz trajectories with clear contrast between nearby starts."""

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Baseline trajectory: scatter coloured by time and a faint spine for context.
    scatter = ax.scatter3D(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=t,
        cmap="viridis",
        s=0.8,
        alpha=0.65,
        label="_nolegend_",
    )
    ax.plot(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        color="0.3",
        linewidth=1.0,
        alpha=0.5,
        label="Baseline",
    )
    fig.colorbar(scatter, ax=ax, label="time (s)")

    # Perturbed trajectory highlighted to emphasise divergence.
    ax.plot(
        X2[:, 0],
        X2[:, 1],
        X2[:, 2],
        color="crimson",
        linewidth=3.0,
        label="$x_2(0)=5.01$ perturbed",
    )
    ax.scatter3D(X[0, 0], X[0, 1], X[0, 2], color="black", s=20, marker="o")
    ax.scatter3D(X2[0, 0], X2[0, 1], X2[0, 2], color="crimson", s=30, marker="^")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("Lorenz sensitivity to perturbing $x_2(0)$")
    ax.view_init(elev=22, azim=-45)
    ax.legend(loc="upper left")

    return fig


__all__ = [
    "ensure_dir",
    "savefig",
    "plot_param_sweep",
    "plot_compare_methods",
    "plot_lorenz_with_timecolor",
]
