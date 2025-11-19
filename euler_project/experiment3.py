"""Experiment set for programming exercise 3 (adaptive step-size control).

Implements an embedded Runge–Kutta method with adaptive steps via a generic
OOP solver. Uses the Bogacki–Shampine 3(2) pair for tasks b)–d).

This module is directly runnable via ``python -m euler_project.experiment3``.
It produces a timestamped run directory under ``euler_project/runs/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
from time import perf_counter

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .integrators import exRungeKutta, adaptive_embedded_rk
from .plotting import (
    ensure_dir,
    savefig,
    plot_adaptive_solution,
    plot_time_grids,
    plot_stepsizes_over_time,
    plot_3d_single,
    plot_multi_approximations,
    plot_stepsize_single,
)
from .problems import (
    rhs_cos2_arctan_problem,
    arctan_analytic,
    rhs_forced_lorenz,
)

Array = np.ndarray


# --- Embedded RK: Bogacki–Shampine 3(2) --------------------------------------


def _butcher_bogacki_shampine_32() -> Tuple[Array, Array, Array, Array, int]:
    """Return A, b_high(3rd), b_low(2nd), c and p_error for BS23.

    The returned ``p_error`` is the order used in the step-size controller
    exponent denominator (1/(p_error+1)). For BS23 we use ``p_error = 2``.
    """

    A = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0 / 2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0 / 4.0, 0.0, 0.0],
            [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0],
        ],
        dtype=float,
    )
    c = np.array([0.0, 0.5, 0.75, 1.0], dtype=float)
    b_high = np.array([2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0], dtype=float)  # order 3
    b_low = np.array([7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0], dtype=float)  # order 2
    p_error = 2
    return A, b_high, b_low, c, p_error


# --- Experiments --------------------------------------------------------------


def run_arctan_problem() -> tuple[Figure, Figure, Figure]:
    """Task (b): adaptive BS23 on the arctan problem with comparisons.

    Parameters as on the sheet: ``T=10``, ``tauMax=0.1``, ``rho=0.9``,
    ``q=2``, ``TOL=1e-3``. Adds a second plot with the adaptive time grid.
    """

    x0 = np.array([np.arctan(-120.0)])
    T = 10.0
    tauMax = 0.1
    rho = 0.9
    q = 2.0
    TOL = 1e-3

    A, b_high, b_low, c, p_error = _butcher_bogacki_shampine_32()

    # Adaptive method (BS23 embedded)
    t_ad, X_ad = adaptive_embedded_rk(
        rhs_cos2_arctan_problem, x0, T, tauMax, rho, q, TOL, A, b_high, b_low, c, p_error
    )

    # Constant-step references for comparison
    t_eu, X_eu = exRungeKutta(rhs_cos2_arctan_problem, x0, T, 0.1, np.array([[0.0]]), np.array([1.0]), np.array([0.0]))
    t_low, X_low = exRungeKutta(rhs_cos2_arctan_problem, x0, T, 0.1, A, b_low, c)
    #fixen nur 1 einmal laufen
    # Exact solution sampled on a fine grid for plotting
    t_exact = np.linspace(0.0, T, 2000)
    x_exact = arctan_analytic(t_exact)

    fig_sol = plot_adaptive_solution(
        t_ad,
        X_ad,
        t_eu,
        X_eu,
        t_low,
        X_low,
        t_exact,
        x_exact,
        title="Arctan problem: adaptive BS23 vs references",
    )

    fig_grid = plot_time_grids([t_ad], ["adaptive grid"], T, title="Adaptive discretization points (BS23)")
    fig_steps = plot_stepsize_single(t_ad, title="Adaptive stepsize")
    return fig_sol, fig_grid, fig_steps


def run_tol_influence() -> tuple[Figure, Figure, Figure, Figure, Figure, dict[str, Array]]:
    """Task (c): effect of tolerance on approximations, grid and stepsizes."""

    x0 = np.array([np.arctan(-120.0)])
    T = 10.0
    tauMax = 0.1
    rho = 0.9
    q = 2.0
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    A, b_high, b_low, c, p_error = _butcher_bogacki_shampine_32()

    grids: list[Array] = []
    labels: list[str] = []
    curves: list[tuple[Array, Array, str]] = []
    err_curves: list[tuple[Array, Array, str]] = []
    runtimes: list[float] = []
    nsteps: list[int] = []
    max_errors: list[float] = []
    final_errors: list[float] = []

    for tol in tolerances:
        t0 = perf_counter()
        t_ad, X_ad = adaptive_embedded_rk(
            rhs_cos2_arctan_problem, x0, T, tauMax, rho, q, tol, A, b_high, b_low, c, p_error
        )
        rt = perf_counter() - t0
        runtimes.append(rt)
        nsteps.append(len(t_ad) - 1)
        grids.append(t_ad)
        label = f"TOL={tol:g}"
        labels.append(label)
        curves.append((t_ad, X_ad, label))
        # errors on the grid
        e = np.abs(X_ad[:, 0] - arctan_analytic(t_ad))
        err_curves.append((t_ad, e, label))
        max_errors.append(float(np.max(e)))
        final_errors.append(float(e[-1]))

    # Exact solution on a fine grid
    t_exact = np.linspace(0.0, T, 2000)
    x_exact = arctan_analytic(t_exact)

    fig_solutions = plot_multi_approximations(curves, t_exact, x_exact, title="Approximations for various TOL")
    fig_grids = plot_time_grids(grids, labels, T, title="Adaptive grids for different tolerances")
    fig_h = plot_stepsizes_over_time(grids, labels, title="Step sizes over time for different TOL")
    # grid error curves
    from .plotting import plot_error_curves, plot_runtime_vs_error, plot_tradeoff_table

    fig_err = plot_error_curves(err_curves, title="Grid error |x_ad - x_exact| for each TOL")
    # accuracy-efficiency trade-off: runtime vs max error
    tols = np.array(tolerances, dtype=float)
    runt_np = np.array(runtimes)
    nsteps_np = np.array(nsteps)
    maxerr_np = np.array(max_errors)
    finalerr_np = np.array(final_errors)
    fig_trade = plot_runtime_vs_error(tols, runt_np, maxerr_np, nsteps_np)
    fig_table = plot_tradeoff_table(tols, runt_np, nsteps_np, maxerr_np, finalerr_np)

    metrics = {
        "tolerances": tols,
        "runtime_sec": np.array(runtimes),
        "n_steps": np.array(nsteps),
        "max_error": np.array(max_errors),
        "final_error": np.array(final_errors),
    }

    return fig_solutions, fig_grids, fig_h, fig_err, fig_trade, {**metrics, "fig_table": fig_table}


def run_lorenz_adaptive() -> tuple[Figure, Figure]:
    """Task (d): forced Lorenz system integrated with adaptive BS23.

    Uses the same forcing as in experiment 2 but now integrated adaptively.
    """

    T = 5.0
    tauMax = 0.1
    rho = 0.9
    q = 2.0
    tol = 1e-3
    x0 = np.array([10.0, 5.0, 12.0])

    f_forced = lambda t, X: rhs_forced_lorenz(t, X, a=10.0, b=25.0, c=8.0 / 3.0, forcing_amplitude=100.0)

    A, b_high, b_low, c, p_error = _butcher_bogacki_shampine_32()
    t, X = adaptive_embedded_rk(f_forced, x0, T, tauMax, rho, q, tol, A, b_high, b_low, c, p_error)

    fig3d = plot_3d_single(t, X, title="Forced Lorenz (adaptive BS23)")
    fig_h = plot_stepsizes_over_time([t], ["adaptive"], title="Adaptive step sizes (Lorenz)")
    return fig3d, fig_h


# ---- CLI helpers -------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent


def _make_run_dirs() -> tuple[Path, Path]:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / f"ex3-{ts}")
    figs_dir = ensure_dir(run_dir / "figs")
    return run_dir, figs_dir


def _create_results_pdf(run_dir: Path, figures: list[Figure]) -> None:
    all_plots_path = run_dir / "all_plots.pdf"
    results_path = run_dir / "results.pdf"

    with PdfPages(all_plots_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    # For this sheet, results.pdf is identical to all_plots.pdf
    with PdfPages(results_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")


def main() -> None:
    run_dir, figs_dir = _make_run_dirs()
    figures: list[Figure] = []

    fig_b_sol, fig_b_grid, fig_b_steps = run_arctan_problem()
    figures.append(fig_b_sol)
    figures.append(fig_b_grid)
    figures.append(fig_b_steps)

    fig_c_sol, fig_c_grid, fig_c_h, fig_c_err, fig_trade, metrics = run_tol_influence()
    figures.append(fig_c_sol)
    figures.append(fig_c_grid)
    figures.append(fig_c_h)
    figures.append(fig_c_err)
    figures.append(fig_trade)
    # add table figure
    if "fig_table" in metrics:
        figures.append(metrics["fig_table"])  # type: ignore[index]

    fig_d_3d, fig_d_h = run_lorenz_adaptive()
    figures.append(fig_d_3d)
    figures.append(fig_d_h)

    # Export
    savefig(fig_b_sol, figs_dir / "b_adaptive_vs_refs")
    savefig(fig_b_grid, figs_dir / "b_adaptive_grid")
    savefig(fig_b_steps, figs_dir / "b_adaptive_stepsize")
    savefig(fig_c_sol, figs_dir / "c_solutions_vs_tol")
    savefig(fig_c_grid, figs_dir / "c_grids_vs_tol")
    savefig(fig_c_h, figs_dir / "c_stepsizes_vs_tol")
    savefig(fig_c_err, figs_dir / "c_grid_errors_vs_tol")
    savefig(fig_trade, figs_dir / "c_accuracy_vs_runtime")
    # save table figure
    if "fig_table" in metrics:
        savefig(metrics["fig_table"], figs_dir / "c_tradeoff_table")  # type: ignore[index]
    savefig(fig_d_3d, figs_dir / "d_lorenz_adaptive_3d")
    savefig(fig_d_h, figs_dir / "d_lorenz_stepsizes")

    _create_results_pdf(run_dir, figures)

    # Close figures
    import matplotlib.pyplot as plt

    for fig in figures:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    print(f"Saved exercise 3 run to: {run_dir}")


if __name__ == "__main__":
    main()
