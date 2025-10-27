"""Experiment set for programming exercise 4 (implicit Euler).

What is implemented here:

- (a) A generic implicit Euler solver is provided in
  :mod:`euler_project.integrators` as :class:`ImplicitEuler` with the wrapper
  :func:`implicitEuler`.
- (c) For the cubic ODE x' = q x - x^3 with q=9, we generate, for each fixed
  point x*, two plots (explicit and implicit Euler) that overlay five
  trajectories started at x*+delta with delta in {-4,-1,0,1,4}.
- (d) Application of implicit Euler to the Lorenz system; we show a 3D
  trajectory for a standard initial value.

This module is directly runnable via

    python -m euler_project.experiment4

and will create a timestamped run directory under ``euler_project/runs/``.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Iterable

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .integrators import explEuler, implicitEuler
from .plotting import (
    ensure_dir,
    savefig,
    plot_family_solutions,
    plot_3d_single,
    plot_lorenz_fixed_points_table,
)
from .problems import rhs_cubic, rhs_lorenz, rhs_forced_lorenz, rhs_forced_lorenz_const

Array = np.ndarray


# ---- (c) Families around fixed points ---------------------------------------


def _family_initial_values(xstar: float) -> list[float]:
    offsets = [-4.0, -1.0, 0.0, 1.0, 4.0]
    return [xstar + d for d in offsets]


def run_fixed_point_families(
    q: float = 9.0, T: float = 2.0, tau: float = 0.05
) -> list[Figure]:
    """Generate six figures as requested in (c).

    For each fixed point x* in {0, ±sqrt(q)} we create two plots: one with the
    explicit Euler trajectories and one with the implicit Euler trajectories.
    """

    f = lambda t, x: rhs_cubic(t, x, q)
    xstars = [0.0, -float(np.sqrt(q)), float(np.sqrt(q))]
    labels = ["x* - 4", "x* - 1", "x*", "x* + 1", "x* + 4"]

    figs: list[Figure] = []
    for xstar in xstars:
        x0_vals = _family_initial_values(xstar)

        # explicit Euler
        Xs_exp: list[Array] = []
        t_ref: Array | None = None
        for x0 in x0_vals:
            t, X = explEuler(f, np.array([x0]), T, tau)
            if t_ref is None:
                t_ref = t
            Xs_exp.append(X)
        assert t_ref is not None
        fig_exp = plot_family_solutions(
            t_ref,
            Xs_exp,
            labels,
            title=f"Cubic (q={q:g}), fixed point x*={xstar:g}: Explicit Euler",
            xstar=xstar,
        )
        figs.append(fig_exp)

        # implicit Euler
        Xs_imp: list[Array] = []
        t_ref2: Array | None = None
        for x0 in x0_vals:
            t2, X2 = implicitEuler(f, np.array([x0]), T, tau)
            if t_ref2 is None:
                t_ref2 = t2
            Xs_imp.append(X2)
        assert t_ref2 is not None
        fig_imp = plot_family_solutions(
            t_ref2,
            Xs_imp,
            labels,
            title=f"Cubic (q={q:g}), fixed point x*={xstar:g}: Implicit Euler",
            xstar=xstar,
        )
        figs.append(fig_imp)

    return figs


# ---- (d) Lorenz with implicit Euler -----------------------------------------


def run_lorenz_implicit(
    x0: Iterable[float] = (10.0, 5.0, 12.0),
    T: float = 50.0,
    tau: float = 1e-3,
    a: float = 10.0,
    b: float = 20.0,
    c: float = 8.0 / 3.0,
) -> Figure:
    """Integrate the unforced Lorenz system with implicit Euler and plot in 3D."""

    x0_arr = np.asarray(x0, dtype=float).reshape(3)
    f = lambda t, X: rhs_lorenz(t, X, a=a, b=b, c=c)
    t, X = implicitEuler(f, x0_arr, T, tau)
    fig3d = plot_3d_single(
        t,
        X,
        title=f"Lorenz trajectory (implicit Euler, a={a}, b={b}, c={c}, unforced)",
    )
    return fig3d


def run_lorenz_forced_const(
    x0: Iterable[float] = (10.0, 5.0, 12.0),
    T: float = 50.0,
    tau: float = 1e-3,
    a: float = 10.0,
    b: float = 20.0,
    c: float = 8.0 / 3.0,
    A: float = 100.0,
) -> Figure:
    x0_arr = np.asarray(x0, dtype=float).reshape(3)
    f = lambda t, X: rhs_forced_lorenz_const(t, X, a=a, b=b, c=c, forcing_amplitude=A)
    t, X = implicitEuler(f, x0_arr, T, tau)
    fig3d = plot_3d_single(
        t,
        X,
        title=f"Lorenz (implicit, const forcing A={A}, a={a}, b={b}, c={c})",
    )
    return fig3d


def run_lorenz_forced_sin(
    x0: Iterable[float] = (10.0, 5.0, 12.0),
    T: float = 50.0,
    tau: float = 1e-3,
    a: float = 10.0,
    b: float = 20.0,
    c: float = 8.0 / 3.0,
    A: float = 100.0,
) -> Figure:
    x0_arr = np.asarray(x0, dtype=float).reshape(3)
    f = lambda t, X: rhs_forced_lorenz(t, X, a=a, b=b, c=c, forcing_amplitude=A)
    t, X = implicitEuler(f, x0_arr, T, tau)
    fig3d = plot_3d_single(
        t,
        X,
        title=f"Lorenz (implicit, sinusoidal forcing A={A}, a={a}, b={b}, c={c})",
    )
    return fig3d


# ---- CLI helpers -------------------------------------------------------------


PACKAGE_DIR = Path(__file__).resolve().parent


def _make_run_dirs() -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / f"ex4-{ts}")
    figs_dir = ensure_dir(run_dir / "figs")
    return run_dir, figs_dir


def _create_results_pdf(run_dir: Path, figures: list[Figure]) -> None:
    all_plots_path = run_dir / "all_plots.pdf"
    results_path = run_dir / "results.pdf"

    with PdfPages(all_plots_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    with PdfPages(results_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")


def main() -> None:
    run_dir, figs_dir = _make_run_dirs()

    figures: list[Figure] = []

    # (c) Families around fixed points (6 figures)
    figs_c = run_fixed_point_families(q=9.0, T=2.0, tau=0.05)
    figures.extend(figs_c)

    # (d) Lorenz with implicit Euler: unforced, constant forcing, sinusoidal forcing
    fig_lorenz_unforced = run_lorenz_implicit()
    fig_lorenz_const = run_lorenz_forced_const()
    fig_lorenz_sin = run_lorenz_forced_sin()
    # Additionally: create a time-series table to highlight fixed points (unforced)
    # Integrate again to retrieve data for sampling
    a, b, c = 10.0, 20.0, 8.0 / 3.0
    x0_arr = np.asarray((10.0, 5.0, 12.0), dtype=float)
    f_unforced = lambda t, X: rhs_lorenz(t, X, a=a, b=b, c=c)
    T_table = 50.0
    tau = 1e-3
    t_grid, X_grid = implicitEuler(f_unforced, x0_arr, T_table, tau)
    # sample a few times
    ts = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    idx = np.clip((ts / tau).astype(int), 0, len(t_grid) - 1)
    Xs = X_grid[idx]
    # equilibria for unforced Lorenz
    S = np.sqrt(c * (b - 1.0))
    equilibria = np.array([[0.0, 0.0, 0.0], [S, S, b - 1.0], [-S, -S, b - 1.0]])
    fig_table = plot_lorenz_fixed_points_table(ts, Xs, equilibria, title="Unforced Lorenz: distances to equilibria over time")
    figures.append(fig_lorenz_unforced)
    figures.append(fig_lorenz_const)
    figures.append(fig_lorenz_sin)
    figures.append(fig_table)

    # Save
    # Families: save with systematic names
    names_c = [
        "c_fixed_x0_explicit",
        "c_fixed_x0_implicit",
        "c_fixed_xminus_explicit",
        "c_fixed_xminus_implicit",
        "c_fixed_xplus_explicit",
        "c_fixed_xplus_implicit",
    ]
    for fig, name in zip(figs_c, names_c):
        savefig(fig, figs_dir / name)
    savefig(fig_lorenz_unforced, figs_dir / "d_lorenz_unforced_3d")
    savefig(fig_lorenz_const,    figs_dir / "d_lorenz_forcing_const_3d")
    savefig(fig_lorenz_sin,      figs_dir / "d_lorenz_forcing_sin_3d")
    savefig(fig_table,           figs_dir / "d_lorenz_fixedpoint_table")

    _create_results_pdf(run_dir, figures)

    # Close
    import matplotlib.pyplot as plt

    for fig in figures:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    print(f"Saved exercise 4 run to: {run_dir}")


if __name__ == "__main__":
    main()
