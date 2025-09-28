"""Experiment set for programming exercise 1 (sheet 2.3).

Provided unchanged to preserve the original functionality. The aggregator
entry point ``euler_project.experiments`` will call these functions together
with those from :mod:`euler_project.experiment2`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

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
    """Task (b): parameter sweep for the cubic ODE."""

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

    return plot_param_sweep(t_ref, trajectories, q_values)


def run_method_comparison(q: float) -> Figure:
    """Task (c): compare Euler against LSODA for a given q."""

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

    return plot_compare_methods(
        t_euler_fine,
        x_euler_fine,
        t_euler_coarse,
        x_euler_coarse,
        t_ref,
        x_ref,
        q,
    )


def run_lorenz_sensitivity() -> tuple[Figure, Figure]:
    """Task (d): sensitivity of the Lorenz system."""

    tau = 0.001
    T = 10.0

    x0 = np.array([10.0, 5.0, 12.0])
    x0_perturbed = np.array([10.0, 5.01, 12.0])

    t, X = explEuler(rhs_lorenz, x0, T, tau)
    t2, X2 = explEuler(rhs_lorenz, x0_perturbed, T, tau)

    fig3d = plot_lorenz_with_timecolor(t, X, t2, X2)
    fig_sep = plot_lorenz_separation(t, X, X2)
    return fig3d, fig_sep


def write_canonical_answers(dest_path: Path) -> None:
    """Write canonical answers (b–d) to `dest_path`."""

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
    """Create a timestamped run directory and its `figs` subfolder."""

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / ts)
    figs_dir = ensure_dir(run_dir / "figs")
    return run_dir, figs_dir


def _create_results_pdf(run_dir: Path, figs_dir: Path, figures: list[Figure]) -> None:
    """Create `all_plots.pdf` and `results.pdf` in `run_dir`.

    - `all_plots.pdf`: concatenation of the raw Matplotlib figures.
    - `results.pdf`: the same plots followed by a final page that renders
      the content of `answers.txt` from the run root.
    """

    all_plots_path = run_dir / "all_plots.pdf"
    results_path = run_dir / "results.pdf"
    answers_path = run_dir / "answers.txt"

    # 1) Concatenate all figures into all_plots.pdf
    with PdfPages(all_plots_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    # 2) Build results.pdf: plots + answers page
    answers_text = ""
    if answers_path.exists():
        try:
            answers_text = answers_path.read_text(encoding="utf-8")
        except Exception:
            answers_text = answers_path.read_text(errors="ignore")
    else:
        answers_text = "answers.txt not found in run directory."

    def _make_answers_page(text: str) -> Figure:
        fig = plt.figure(figsize=(8.5, 11.0))
        fig.suptitle("Answers", fontsize=16, y=0.98)
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
        ax.axis("off")
        ax.text(
            0.0,
            1.0,
            text,
            va="top",
            ha="left",
            family="monospace",
            fontsize=10.5,
            transform=ax.transAxes,
            wrap=True,
        )
        return fig

    with PdfPages(results_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
        ans_fig = _make_answers_page(answers_text)
        try:
            pdf.savefig(ans_fig, bbox_inches="tight")
        finally:
            if plt.fignum_exists(ans_fig.number):
                plt.close(ans_fig)


def main() -> None:
    """Run all experiments and export figures, answers.txt, and combined PDFs.

    - Saves individual figures (PNG+PDF) under the run's `figs/` folder.
    - Writes a canonical `answers.txt` to the run root.
    - Creates `all_plots.pdf` (all figures) and `results.pdf` (figures + answers).
    """

    run_dir, figs_dir = _make_run_dirs()

    figures: list[Figure] = []

    # (b) Parameter sweep
    fig_param = run_parameter_study()
    figures.append(fig_param)

    # (c) Method comparisons for q=10 and q=0.1
    fig_q10 = run_method_comparison(10.0)
    figures.append(fig_q10)

    fig_q01 = run_method_comparison(0.1)
    figures.append(fig_q01)

    # (d) Lorenz sensitivity (3D + separation)
    fig_lorenz, fig_lorenz_sep = run_lorenz_sensitivity()
    figures.append(fig_lorenz)
    figures.append(fig_lorenz_sep)

    # Export individual figures
    savefig(fig_param, figs_dir / "param_sweep_cubic.png")
    savefig(fig_q10, figs_dir / "compare_q10.png")
    savefig(fig_q01, figs_dir / "compare_q01.png")
    savefig(fig_lorenz, figs_dir / "lorenz_sensitivity.png")
    savefig(fig_lorenz_sep, figs_dir / "lorenz_separation.png")

    # Write canonical answers
    write_canonical_answers(run_dir / "answers.txt")

    # Create combined PDFs before closing figures
    _create_results_pdf(run_dir, figs_dir, figures)

    # Close open figures to free resources
    for fig in figures:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
