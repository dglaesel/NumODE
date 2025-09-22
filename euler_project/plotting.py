"""Plotting utilities for the Euler ODE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection

Array = np.ndarray


FIGS_DIR = Path("figs")


def ensure_dir(path: Path | str) -> Path:
    """Ensure that the directory exists and return its path object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def savefig(fig: Figure, filepath: Path | str) -> None:
    """Save a Matplotlib figure with tight layout."""

    fig.tight_layout()
    fig.savefig(filepath, dpi=300)


def plot_param_sweep(t: Array, trajectories: Sequence[Array], qs: Sequence[float]) -> Figure:
    """Plot multiple trajectories from the cubic parameter sweep."""

    fig, ax = plt.subplots(figsize=(8, 5))
    for X, q in zip(trajectories, qs):
        ax.plot(t, X[:, 0], label=f"q = {q}")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Cubic ODE parameter sweep")
    ax.grid(True)
    ax.legend()
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
    ax.plot(t_ref, x_ref[:, 0], label="LSODA (reference)", linewidth=2)
    ax.plot(t_euler_fine, x_euler_fine[:, 0], label="Euler τ=0.01", linestyle="--")
    ax.plot(t_euler_coarse, x_euler_coarse[:, 0], label="Euler τ=0.1", linestyle=":")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(f"Method comparison for q = {q}")
    ax.grid(True)
    ax.legend()
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
        label="Baseline (time-coloured)",
    )
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color="black", linewidth=0.5, alpha=0.4)
    fig.colorbar(scatter, ax=ax, label="time")

    # Perturbed trajectory highlighted to emphasise divergence.
    ax.plot(
        X2[:, 0],
        X2[:, 1],
        X2[:, 2],
        color="crimson",
        linewidth=1.4,
        label="$x_2(0)=5.01$ perturbed",
    )
    ax.scatter3D(X[0, 0], X[0, 1], X[0, 2], color="black", s=20, marker="o")
    ax.scatter3D(X2[0, 0], X2[0, 1], X2[0, 2], color="crimson", s=30, marker="^")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("Lorenz sensitivity to perturbing $x_2(0)$")
    ax.legend(loc="upper left")

    return fig


__all__ = [
    "ensure_dir",
    "savefig",
    "plot_param_sweep",
    "plot_compare_methods",
    "plot_lorenz_with_timecolor",
]
