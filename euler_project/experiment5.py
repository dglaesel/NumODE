"""Experiment set for programming exercise 5 (implicit RK + isometries).

What is implemented:

- (1) A generic implicit Runge-Kutta solver lives in
  :mod:`euler_project.integrators` as :class:`ImplicitRungeKutta` with the
  wrappers :func:`implicitRungeKutta` and :func:`implicitRK`.
- (2) Stability test on the cubic ODE x' = q x - x^3 using four methods:
  explicit Euler, implicit Euler, trapezoidal rule (2-stage IRK), implicit
  midpoint rule; a zoomed-in variant is also provided.
- (3) Phase plots for undamped and damped harmonic oscillators comparing the
  same four methods.
- (4) Consistency-order study for the undamped oscillator via a log2-log2
  plot; slopes are estimated by linear least squares.

This module is runnable via

    python -m euler_project.experiment5

and creates a timestamped run directory under ``euler_project/runs/``.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Iterable, Tuple

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .integrators import explEuler, implicitRungeKutta, implicitRK
from .plotting import (
    ensure_dir,
    savefig,
    plot_family_solutions,
    plot_convergence,
    plot_task2_approximations_1d,
    plot_task3_phase,
)
from .problems import rhs_cubic, rhs_harmonic_oscillator, oscillator_exact_undamped

Array = np.ndarray


# ---- Butcher arrays (implicit) ----------------------------------------------


def _butcher_explicit_euler() -> Tuple[Array, Array, Array]:
    A = np.array([[0.0]])
    b = np.array([1.0])
    c = np.array([0.0])
    return A, b, c


def _butcher_implicit_euler() -> Tuple[Array, Array, Array]:
    A = np.array([[1.0]])
    b = np.array([1.0])
    c = np.array([1.0])
    return A, b, c


def _butcher_trapezoidal() -> Tuple[Array, Array, Array]:
    # As specified: s=2, c=(0,1), b=(1/2,1/2), A=[[0,0],[1/2,1/2]]
    A = np.array([[0.0, 0.0], [0.5, 0.5]])
    b = np.array([0.5, 0.5])
    c = np.array([0.0, 1.0])
    return A, b, c


def _butcher_implicit_midpoint() -> Tuple[Array, Array, Array]:
    A = np.array([[0.5]])
    b = np.array([1.0])
    c = np.array([0.5])
    return A, b, c


# ---- (2) Stability behaviour on cubic ODE -----------------------------------


def run_stability_cubic(
    q: float = 25.0, T: float = 5.0, tau: float = 0.05, x0: float = 3.0
) -> tuple[Figure, Figure]:
    f = lambda t, x: rhs_cubic(t, x, q)
    x0_arr = np.array([x0])

    # Methods
    Aee, bee, cee = _butcher_explicit_euler()
    Aie, bie, cie = _butcher_implicit_euler()
    Atr, btr, ctr = _butcher_trapezoidal()
    Amp, bmp, cmp = _butcher_implicit_midpoint()

    t_eu, X_eu = explEuler(f, x0_arr, T, tau)
    # Use IRK for the three implicit methods for a uniform interface
    t_ie, X_ie = implicitRungeKutta(f, x0_arr, T, tau, Aie, bie, cie, solver="fsolve")
    t_tr, X_tr = implicitRungeKutta(f, x0_arr, T, tau, Atr, btr, ctr, solver="fsolve")
    t_mp, X_mp = implicitRungeKutta(f, x0_arr, T, tau, Amp, bmp, cmp, solver="fsolve")

    labels = ["explicit Euler", "implicit Euler", "trapezoidal", "midpoint"]
    # All share the same grid
    fig_all = plot_task2_approximations_1d(t_eu, [X_eu, X_ie, X_tr, X_mp], labels)

    # Zoomed view
    fig_zoom = plot_task2_approximations_1d(
        t_eu, [X_eu, X_ie, X_tr, X_mp], labels, title="Different approximations - 1D"
    )
    # Set x-limits to a representative interval
    if fig_zoom.axes:
        ax = fig_zoom.axes[0]
        ax.set_xlim(1.0, 1.8)

    return fig_all, fig_zoom


# ---- (3) Phase plots for (damped) harmonic oscillator -----------------------


def run_oscillator_phase(
    alpha: float, T: float = 10.0, tau: float = 0.05, x0: Iterable[float] = (1.0, 0.0)
) -> Figure:
    f = lambda t, x: rhs_harmonic_oscillator(t, x, alpha)
    x0_arr = np.asarray(x0, dtype=float)

    Aie, bie, cie = _butcher_implicit_euler()
    Atr, btr, ctr = _butcher_trapezoidal()
    Amp, bmp, cmp = _butcher_implicit_midpoint()

    t_eu, X_eu = explEuler(f, x0_arr, T, tau)
    t_ie, X_ie = implicitRK(f, x0_arr, T, tau, Aie, bie, cie, solver="fsolve")
    t_tr, X_tr = implicitRK(f, x0_arr, T, tau, Atr, btr, ctr, solver="fsolve")
    t_mp, X_mp = implicitRK(f, x0_arr, T, tau, Amp, bmp, cmp, solver="fsolve")

    labels = ["explicit Euler", "implicit Euler", "trapezoidal", "midpoint"]
    return plot_task3_phase([X_eu, X_ie, X_tr, X_mp], labels, title=(
        f"Harmonic Oscillator - 2D (alpha={alpha:g})"
    ))


# ---- (4) Consistency order via log2-log2 plot -------------------------------


def run_order_study(
    T: float = 5.0, ks: Iterable[int] = range(3, 11)
) -> Figure:
    # Undamped oscillator exact solution
    x0 = np.array([1.0, 0.0])
    Aie, bie, cie = _butcher_implicit_euler()
    Atr, btr, ctr = _butcher_trapezoidal()
    Amp, bmp, cmp = _butcher_implicit_midpoint()

    taus = 2.0 ** (-np.asarray(list(ks), dtype=float))
    err_eu: list[float] = []
    err_ie: list[float] = []
    err_tr: list[float] = []
    err_mp: list[float] = []

    xT_exact = oscillator_exact_undamped(T)
    xT_exact = np.asarray(xT_exact, dtype=float).reshape(2)

    for h in taus:
        f = lambda t, x: rhs_harmonic_oscillator(t, x, alpha=0.0)
        _, X_eu = explEuler(f, x0, T, float(h))
        _, X_ie = implicitRungeKutta(f, x0, T, float(h), Aie, bie, cie, solver="fsolve")
        _, X_tr = implicitRungeKutta(f, x0, T, float(h), Atr, btr, ctr, solver="fsolve")
        _, X_mp = implicitRungeKutta(f, x0, T, float(h), Amp, bmp, cmp, solver="fsolve")
        err_eu.append(float(np.linalg.norm(X_eu[-1] - xT_exact)))
        err_ie.append(float(np.linalg.norm(X_ie[-1] - xT_exact)))
        err_tr.append(float(np.linalg.norm(X_tr[-1] - xT_exact)))
        err_mp.append(float(np.linalg.norm(X_mp[-1] - xT_exact)))

    errors = {
        "explicit Euler": np.array(err_eu),
        "implicit Euler": np.array(err_ie),
        "trapezoidal": np.array(err_tr),
        "midpoint": np.array(err_mp),
    }
    return plot_convergence(taus, errors)


# ---- CLI helpers -------------------------------------------------------------


PACKAGE_DIR = Path(__file__).resolve().parent


def _make_run_dirs() -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / f"ex5-{ts}")
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

    # (2) cubic stability
    fig_all, fig_zoom = run_stability_cubic()
    figures.extend([fig_all, fig_zoom])

    # (3) oscillator phase plots
    fig_undamped = run_oscillator_phase(alpha=0.0)
    fig_damped = run_oscillator_phase(alpha=0.02)
    figures.extend([fig_undamped, fig_damped])

    # (4) order study
    fig_order = run_order_study()
    figures.append(fig_order)

    # Save individual figures
    savefig(fig_all, figs_dir / "task2_approximations_1d")
    savefig(fig_zoom, figs_dir / "task2_approximations_1d_zoom")
    savefig(fig_undamped, figs_dir / "task3_phase_undamped")
    savefig(fig_damped, figs_dir / "task3_phase_damped")
    savefig(fig_order, figs_dir / "task4_convergence_order")

    _create_results_pdf(run_dir, figures)

    # Close
    import matplotlib.pyplot as plt

    for fig in figures:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    print(f"Saved exercise 5 run to: {run_dir}")


__all__ = [
    "run_stability_cubic",
    "run_oscillator_phase",
    "run_order_study",
]


if __name__ == "__main__":
    main()
