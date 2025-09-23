"""Experiment runners for the explicit Euler project."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

import numpy as np
from scipy.integrate import solve_ivp

from .integrators import explEuler
from .plotting import (
    ensure_dir,
    plot_compare_methods,
    plot_lorenz_with_timecolor,
    plot_param_sweep,
    savefig,
)
from .problems import rhs_cubic, rhs_lorenz

Array = np.ndarray

PACKAGE_DIR = Path(__file__).resolve().parent


def run_parameter_study() -> Figure:
    """Run the cubic ODE parameter sweep and return the generated figure."""

    q_values = [0.1, 0.5, 1.0, 2.0, 10.0, 20.0]
    x0 = np.array([2.0])
    tau = 0.01
    T = 10.0

    trajectories: list[Array] = []
    t_ref: Array | None = None

    for q in q_values:
        cubic = lambda t, x, q=q: rhs_cubic(t, x, q)
        t, X = explEuler(cubic, x0, T, tau)
        trajectories.append(X)
        if t_ref is None:
            t_ref = t

    if t_ref is None:
        raise RuntimeError("Time grid was not generated during parameter sweep.")

    fig = plot_param_sweep(t_ref, trajectories, q_values)
    return fig


def run_method_comparison(q: float) -> Figure:
    """Compare Euler solutions with an LSODA reference for a given q."""

    x0 = np.array([2.0])
    T = 10.0

    cubic = lambda t, x, q=q: rhs_cubic(t, x, q)

    t_euler_fine, x_euler_fine = explEuler(cubic, x0, T, 0.01)
    t_euler_coarse, x_euler_coarse = explEuler(cubic, x0, T, 0.1)

    sol = solve_ivp(
        rhs_cubic,
        (0.0, T),
        x0,
        args=(q,),
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    t_ref = sol.t
    x_ref = sol.y.T

    fig = plot_compare_methods(
        t_euler_fine,
        x_euler_fine,
        t_euler_coarse,
        x_euler_coarse,
        t_ref,
        x_ref,
        q,
    )
    return fig


def run_lorenz_sensitivity() -> Figure:
    """Simulate the Lorenz system for two nearby initial conditions."""

    tau = 0.001
    T = 10.0

    x0 = np.array([10.0, 5.0, 12.0])
    x0_perturbed = np.array([10.0, 5.01, 12.0])

    t, X = explEuler(rhs_lorenz, x0, T, tau)
    t2, X2 = explEuler(rhs_lorenz, x0_perturbed, T, tau)

    fig = plot_lorenz_with_timecolor(t, X, t2, X2)
    return fig


def write_answers_template_to(dest_path: Path) -> None:
    """Create a blank answer template for manual completion at dest_path."""

    text = "\n".join(
        [
            "(b) Parameter study – qualitative long-term behaviour vs q:",
            "",
            "",
            "(c, q=10) Method comparison – Euler vs LSODA:",
            "",
            "",
            "(c, q=0.1) Method comparison – Euler vs LSODA:",
            "",
            "",
            "(d) Lorenz sensitivity – describe whether/when trajectories diverge:",
            "",
            "",
        ]
    )
    dest_path.write_text(text, encoding="utf-8")


def _make_run_dirs() -> tuple[Path, Path]:
    """Create a timestamped run directory and its figs subfolder.

    Returns (run_dir, figs_dir).
    """

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / ts)
    figs_dir = ensure_dir(run_dir / "figs")
    return run_dir, figs_dir


def main() -> None:
    """Run all experiments, export figures, and rewrite the answer summary."""

    run_dir, figs_dir = _make_run_dirs()

    figures: list[Figure] = []

    fig_param = run_parameter_study()
    figures.append(fig_param)

    fig_q10 = run_method_comparison(10.0)
    figures.append(fig_q10)

    fig_q01 = run_method_comparison(0.1)
    figures.append(fig_q01)

    fig_lorenz = run_lorenz_sensitivity()
    figures.append(fig_lorenz)

    # Save individual PNGs for this run.
    savefig(fig_param, figs_dir / "param_sweep_cubic.png")
    savefig(fig_q10, figs_dir / "compare_q10.png")
    savefig(fig_q01, figs_dir / "compare_q01.png")
    savefig(fig_lorenz, figs_dir / "lorenz_sensitivity.png")

    # Also save a combined PDF of all figures in the run root.
    pdf_path = run_dir / "all_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")

    # Provide a blank answer template for manual completion.
    write_answers_template_to(run_dir / "answers.txt")

    # Close figures after exporting to avoid resource warnings.
    for fig in figures:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    # Print output location for convenience when running from CLI.
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
