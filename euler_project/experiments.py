"""Experiment suite and CLI entry point for the project.

This module orchestrates the numerical experiments for the sheet 2.3
assignment and is the only module you need to execute directly
(`python -m euler_project.experiments`). It does not implement numerical
methods itself â€“ those live in dedicated modules â€“ but wires together:

- the problems to solve (problems.py)
- the explicit Euler integrator (integrators.py)
- the plotting helpers (plotting.py)

For every run it creates a timestamped folder under
`euler_project/runs/<YYYYMMDD-HHMMSS>/` that contains:
- `figs/`: all figures exported as PNG and PDF
- `all_plots.pdf`: concatenation of the Matplotlib figure objects
- `answers.txt`: a blank template you can fill manually
- `results.pdf`: plots first, your answers page appended at the end

The experiments cover tasks (b)â€“(d) of the sheet:
- (b) Parameter sweep for the scalar cubic ODE
- (c) Method comparison (Euler vs. LSODA) for q=10 and q=0.1
- (d) Lorenz system sensitivity with a small perturbation
"""

from __future__ import annotations  # postpone evaluation of type hints (forward refs)

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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
from .answers import ANSWERS

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

    # Visualise both the 3D trajectories and |Î”X|(t) on a log scale
    fig3d = plot_lorenz_with_timecolor(t, X, t2, X2)
    fig_sep = plot_lorenz_separation(t, X, X2)
    return fig3d, fig_sep


def write_canonical_answers(dest_path: Path) -> None:
    """Write the canonical answers (bâ€“d) to `dest_path`.

    We preserve the original LaTeX-ish markup in the file; the PDF builder
    below sanitizes it for robust rendering.
    """

    sections = [
        ("(b) Long-term behaviour as a function of q", ANSWERS["b"]),
        ("(c) Method comparison (Euler vs. LSODA) and effect of q", ANSWERS["c"]),
        ("(d) Sensitivity for the Lorenz system", ANSWERS["d"]),
    ]
    lines: list[str] = []
    for title, body in sections:
        lines.append(title)
        lines.append("")
        lines.append(body.strip())
        lines.append("")
        lines.append("")
    dest_path.write_text("\n".join(lines), encoding="utf-8")


def _make_run_dirs() -> tuple[Path, Path]:
    """Create a timestamped run directory and its `figs` subfolder.

    Returns a pair (`run_dir`, `figs_dir`).
    """

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / ts)
    figs_dir = ensure_dir(run_dir / "figs")
    return run_dir, figs_dir


def _create_results_pdf(run_dir: Path, figs_dir: Path) -> None:
    """PDF collation disabled: user will create PDFs externally."""

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

    return None\r\n\r\n# -- PDF/LaTeX generation removed per user request (write results.tex and compile if TeX is present) ---






