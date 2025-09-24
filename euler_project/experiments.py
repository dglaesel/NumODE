"""Experiment suite and CLI entry point for the project.

This module orchestrates the numerical experiments for the sheet 2.3
assignment and is the only module you need to execute directly
(`python -m euler_project.experiments`). It does not implement numerical
methods itself – those live in dedicated modules – but wires together:

- the problems to solve (problems.py)
- the explicit Euler integrator (integrators.py)
- the plotting helpers (plotting.py)

For every run it creates a timestamped folder under
`euler_project/runs/<YYYYMMDD-HHMMSS>/` that contains:
- `figs/`: all figures exported as PNG and PDF
- `all_plots.pdf`: concatenation of the Matplotlib figure objects
- `answers.txt`: a blank template you can fill manually
- `results.pdf`: plots first, your answers page appended at the end

The experiments cover tasks (b)–(d) of the sheet:
- (b) Parameter sweep for the scalar cubic ODE
- (c) Method comparison (Euler vs. LSODA) for q=10 and q=0.1
- (d) Lorenz system sensitivity with a small perturbation
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib.figure import Figure

import numpy as np
from scipy.integrate import solve_ivp

from .integrators import explEuler
from .plotting import (
    ensure_dir,
    plot_compare_methods,
    plot_lorenz_with_timecolor,
    plot_lorenz_separation,
    plot_param_sweep,
    savefig,
)
from .problems import rhs_cubic, rhs_lorenz

Array = np.ndarray

# Package root (euler_project/)
PACKAGE_DIR = Path(__file__).resolve().parent


def run_parameter_study() -> Figure:
    """Task (b): parameter sweep for the cubic ODE.

    Simulates the scalar IVP x' = q x - x^3 for several q values using the
    explicit Euler wrapper and returns the Figure that overlays all
    trajectories and their equilibria.
    """

    # Experiment configuration for task (b)
    q_values = [0.1, 0.5, 1.0, 2.0, 10.0, 20.0]
    x0 = np.array([2.0])  # initial condition
    tau = 0.01            # Euler step size
    T = 10.0              # final time

    trajectories: list[Array] = []
    t_ref: Array | None = None

    for q in q_values:
        # Bind q for this loop iteration and advance with explicit Euler
        cubic = lambda t, x, q=q: rhs_cubic(t, x, q)
        t, X = explEuler(cubic, x0, T, tau)
        trajectories.append(X)
        if t_ref is None:
            t_ref = t

    if t_ref is None:
        raise RuntimeError("Time grid was not generated during parameter sweep.")

    # Render a single figure containing all trajectories
    fig = plot_param_sweep(t_ref, trajectories, q_values)
    return fig


def run_method_comparison(q: float) -> Figure:
    """Task (c): compare Euler against LSODA for a given q.

    Runs Euler with two step sizes (0.01 and 0.1) and contrasts them against
    a high-accuracy SciPy `solve_ivp(..., method='LSODA')` reference.
    Returns a Figure that shows both the solution and absolute error bands.
    """

    x0 = np.array([2.0])  # same initial value as in (b)
    T = 10.0

    cubic = lambda t, x, q=q: rhs_cubic(t, x, q)

    t_euler_fine, x_euler_fine = explEuler(cubic, x0, T, 0.01)
    t_euler_coarse, x_euler_coarse = explEuler(cubic, x0, T, 0.1)

    # High-accuracy reference solution used for comparison
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


def run_lorenz_sensitivity() -> tuple[Figure, Figure]:
    """Task (d): sensitivity of the Lorenz system.

    Simulates the Lorenz-63 system from two nearby initial conditions.
    Returns a pair (3D trajectory Figure, 2D separation Figure).
    """

    tau = 0.001  # small step to resolve the dynamics
    T = 10.0

    x0 = np.array([10.0, 5.0, 12.0])
    x0_perturbed = np.array([10.0, 5.01, 12.0])

    t, X = explEuler(rhs_lorenz, x0, T, tau)
    t2, X2 = explEuler(rhs_lorenz, x0_perturbed, T, tau)

    # Visualise both the 3D trajectories and |ΔX|(t) on a log scale
    fig3d = plot_lorenz_with_timecolor(t, X, t2, X2)
    fig_sep = plot_lorenz_separation(t, X, X2)
    return fig3d, fig_sep


def write_answers_template_to(dest_path: Path) -> None:
    """Create a blank answer template for manual completion at `dest_path`.

    The template contains the question prompts for (b)–(d) so you can type
    your discussion right in the generated text file. The combined
    `results.pdf` will append whatever is in this file at the end.
    """

    text = "\n".join(
        [
            "(b) Parameter study - qualitative long-term behaviour vs q:",
            "",
            "",
            "(c, q=10) Method comparison - Euler vs LSODA:",
            "",
            "",
            "(c, q=0.1) Method comparison - Euler vs LSODA:",
            "",
            "",
            "(d) Lorenz sensitivity - describe whether/when trajectories diverge:",
            "",
            "",
        ]
    )
    dest_path.write_text(text, encoding="utf-8")


def _make_run_dirs() -> tuple[Path, Path]:
    """Create a timestamped run directory and its `figs` subfolder.

    Returns a pair (`run_dir`, `figs_dir`).
    """

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / ts)
    figs_dir = ensure_dir(run_dir / "figs")
    return run_dir, figs_dir


def _create_results_pdf(run_dir: Path, figs_dir: Path) -> None:
    """Build `results.pdf` with plots first and answers at the end.

    The figure pages are taken from the exported PNGs to guarantee WYSIWYG
    equality with what you see on disk. The last page embeds the content of
    `answers.txt` if present.
    """

    results_pdf = run_dir / "results.pdf"
    answers_path = run_dir / "answers.txt"

    # Page order of the combined results
    order = [
        "param_sweep_cubic",
        "compare_q10",
        "compare_q01",
        "lorenz_sensitivity",
        "lorenz_separation",
    ]

    with PdfPages(results_pdf) as pdf:
        # Add figure pages from the exported PNGs
        for name in order:
            png = figs_dir / f"{name}.png"
            if not png.exists():
                continue
            img = mpimg.imread(png)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))  # landscape A4-ish
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Append answers page if available
        if answers_path.exists():
            text = answers_path.read_text(encoding="utf-8", errors="ignore")
            fig = plt.figure(figsize=(8.5, 11))  # portrait
            fig.text(0.06, 0.95, text, va="top", ha="left", fontsize=11)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    """Entry point used by `python -m euler_project.experiments`.

    Runs all experiments, exports individual plots, writes the answer
    template, and generates the combined `results.pdf` for the run.
    """

    run_dir, figs_dir = _make_run_dirs()

    figures: list[Figure] = []  # keep open figures to close them later

    # Task (b)
    fig_param = run_parameter_study()
    figures.append(fig_param)

    # Task (c): q=10 and q=0.1
    fig_q10 = run_method_comparison(10.0)
    figures.append(fig_q10)

    fig_q01 = run_method_comparison(0.1)
    figures.append(fig_q01)

    # Task (d)
    fig_lorenz, fig_lorenz_sep = run_lorenz_sensitivity()
    figures.append(fig_lorenz)
    figures.append(fig_lorenz_sep)

    # Save individual PNG/PDFs for this run
    savefig(fig_param, figs_dir / "param_sweep_cubic.png")
    savefig(fig_q10, figs_dir / "compare_q10.png")
    savefig(fig_q01, figs_dir / "compare_q01.png")
    savefig(fig_lorenz, figs_dir / "lorenz_sensitivity.png")
    savefig(fig_lorenz_sep, figs_dir / "lorenz_separation.png")

    # Save a concatenation of the raw Matplotlib figures (diagnostic)
    pdf_path = run_dir / "all_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            # Avoid tight_layout here (3D axes are incompatible); rely on bbox_inches
            pdf.savefig(fig, bbox_inches="tight")

    # Provide a blank answer template for manual completion
    write_answers_template_to(run_dir / "answers.txt")

    # Build the user-facing results.pdf (plots first, answers last)
    _create_results_pdf(run_dir, figs_dir)

    # Close figures after exporting to avoid resource warnings
    for fig in figures:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    # Print output location for convenience when running from CLI
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()

