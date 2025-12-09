"""Experiment set for programming exercise 2 (Runge-Kutta methods).

Provides logistic ODE comparisons, a convergence study, and a forced Lorenz
simulation using the midpoint RK method versus explicit Euler.

This module is directly runnable via ``python -m euler_project.experiment2``.
It produces a timestamped run directory under ``euler_project/runs/``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from .integrators import exRungeKutta, explEuler
from .plotting import (
    plot_convergence,
    plot_forced_lorenz,
    plot_logistic_comparison,
    ensure_dir,
    savefig,
    plot_lorenz_difference,
)
from .problems import logistic_analytic, rhs_forced_lorenz, rhs_logistic

Array = np.ndarray


# --- Butcher arrays -------------------------------------------------------


def _butcher_euler() -> Tuple[Array, Array, Array]:
    A = np.array([[0.0]], dtype=float)
    b = np.array([1.0], dtype=float)
    c = np.array([0.0], dtype=float)
    return A, b, c


def _butcher_midpoint() -> Tuple[Array, Array, Array]:
    A = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=float)
    b = np.array([0.0, 1.0], dtype=float)
    c = np.array([0.0, 0.5], dtype=float)
    return A, b, c


def _butcher_rk4() -> Tuple[Array, Array, Array]:
    A = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    b = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=float)
    c = np.array([0.0, 0.5, 0.5, 1.0], dtype=float)
    return A, b, c


# --- Experiments ----------------------------------------------------------


def run_logistic_methods(q: float) -> Figure:
    """Integrate the logistic ODE with three methods and compare to analytic.

    Uses ``x(0) = 2``, ``T = 10``, ``tau = 0.5``.
    """

    x0 = np.array([2.0])
    T = 10.0
    tau = 0.5

    f = lambda t, x, q=q: rhs_logistic(t, x, q)

    A1, b1, c1 = _butcher_euler()
    A2, b2, c2 = _butcher_midpoint()
    A4, b4, c4 = _butcher_rk4()

    t1, X1 = exRungeKutta(f, x0, T, tau, A1, b1, c1)
    t2, X2 = exRungeKutta(f, x0, T, tau, A2, b2, c2)
    t4, X4 = exRungeKutta(f, x0, T, tau, A4, b4, c4)

    # All use the same uniform grid, so choose one
    t = t1
    exact = logistic_analytic(t, float(x0[0]), q)

    # Labels styled to match the sample figures
    labels = ["explicit Euler", "Runge Method", "classical Runge Method"]
    return plot_logistic_comparison(t, [X1, X2, X4], exact, labels, q)


def run_convergence_study() -> Figure:
    """Convergence for the logistic ODE at ``T = 2`` with ``q = 1``.

    Runs Euler, midpoint and RK4 with ``tau = 2^{-k}``, for ``k = 3..10``.
    The error is measured at ``T = 2`` against the analytic solution.
    """

    q = 1.0
    T = 2.0
    x0 = np.array([2.0])

    ks = np.arange(3, 11)
    taus = 2.0 ** (-ks)

    A1, b1, c1 = _butcher_euler()
    A2, b2, c2 = _butcher_midpoint()
    A4, b4, c4 = _butcher_rk4()

    err_euler: list[float] = []
    err_mid: list[float] = []
    err_rk4: list[float] = []

    xT_exact = float(logistic_analytic(T, float(x0[0]), q))

    for tau in taus:
        f = lambda t, x, q=q: rhs_logistic(t, x, q)
        _, X1 = exRungeKutta(f, x0, T, float(tau), A1, b1, c1)
        _, X2 = exRungeKutta(f, x0, T, float(tau), A2, b2, c2)
        _, X4 = exRungeKutta(f, x0, T, float(tau), A4, b4, c4)
        err_euler.append(abs(X1[-1, 0] - xT_exact))
        err_mid.append(abs(X2[-1, 0] - xT_exact))
        err_rk4.append(abs(X4[-1, 0] - xT_exact))

    errors = {"Euler": np.array(err_euler), "midpoint": np.array(err_mid), "RK4": np.array(err_rk4)}
    return plot_convergence(taus, errors)


def run_forced_lorenz() -> tuple[Figure, Figure]:
    """Forced Lorenz system via midpoint RK vs explicit Euler."""

    T = 5.0
    tau = 0.001
    x0 = np.array([10.0, 5.0, 12.0])

    A2, b2, c2 = _butcher_midpoint()

    f_forced = lambda t, X: rhs_forced_lorenz(t, X, a=10.0, b=25.0, c=8.0 / 3.0, forcing_amplitude=100.0)

    t_rk, X_rk = exRungeKutta(f_forced, x0, T, tau, A2, b2, c2)
    t_eu, X_eu = explEuler(f_forced, x0, T, tau)

    fig3d = plot_forced_lorenz(t_rk, X_rk, t_eu, X_eu)
    figdiff = plot_lorenz_difference(t_rk, X_rk, t_eu, X_eu)
    return fig3d, figdiff


__all__ = [
    "run_logistic_methods",
    "run_convergence_study",
    "run_forced_lorenz",
]


# ---- CLI helpers ----------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent


def _make_run_dirs() -> tuple[Path, Path]:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / f"ex2-{ts}")
    figs_dir = ensure_dir(run_dir / "figs")
    return run_dir, figs_dir


def _create_results_pdf(run_dir: Path, figures: list[Figure]) -> None:
    all_plots_path = run_dir / "all_plots.pdf"
    results_path = run_dir / "results.pdf"

    with PdfPages(all_plots_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    # results.pdf identical to all_plots for exercise 2
    with PdfPages(results_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")


def main() -> None:
    run_dir, figs_dir = _make_run_dirs()

    figures: list[Figure] = []

    fig_log_q01 = run_logistic_methods(0.1)
    figures.append(fig_log_q01)
    fig_log_q1 = run_logistic_methods(1.0)
    figures.append(fig_log_q1)
    fig_conv = run_convergence_study()
    figures.append(fig_conv)
    fig_forced, fig_diff = run_forced_lorenz()
    figures.append(fig_forced)
    figures.append(fig_diff)

    savefig(fig_log_q01, figs_dir / "logistic_q0p1")
    savefig(fig_log_q1, figs_dir / "logistic_q1")
    savefig(fig_conv, figs_dir / "convergence_logistic")
    savefig(fig_forced, figs_dir / "forced_lorenz_midpoint_vs_euler")
    savefig(fig_diff, figs_dir / "forced_lorenz_difference")

    _create_results_pdf(run_dir, figures)

    for fig in figures:
        import matplotlib.pyplot as plt

        if plt.fignum_exists(fig.number):
            plt.close(fig)

    print(f"Saved exercise 2 run to: {run_dir}")


if __name__ == "__main__":
    main()
