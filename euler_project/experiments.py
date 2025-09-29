"""Aggregator entry point for programming exercises 1 and 2.

Runs all original figures from experiment 1 and the new Runge–Kutta
experiments from experiment 2. Produces a timestamped run directory with
figures, an `answers.txt` for exercise 1, and composite PDFs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .plotting import ensure_dir, savefig
from .experiment1 import (
    run_parameter_study,
    run_method_comparison,
    run_lorenz_sensitivity,
    write_canonical_answers,
)
from .experiment2 import run_convergence_study, run_forced_lorenz, run_logistic_methods


# Package root (euler_project/)
PACKAGE_DIR = Path(__file__).resolve().parent


def _make_run_dirs() -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(PACKAGE_DIR / "runs" / ts)
    figs_dir = ensure_dir(run_dir / "figs")
    return run_dir, figs_dir


def _create_results_pdf(run_dir: Path, figures: list[Figure]) -> None:
    all_plots_path = run_dir / "all_plots.pdf"
    results_path = run_dir / "results.pdf"
    answers_path = run_dir / "answers.txt"

    # 1) Concatenate all figures into all_plots.pdf
    with PdfPages(all_plots_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    # 2) Build results.pdf with answers page appended (exercise 1 answers)
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
        ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=10.5, transform=ax.transAxes, wrap=True)
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
    run_dir, figs_dir = _make_run_dirs()

    figures: list[Figure] = []

    # ---- Exercise 1 figures ----
    fig_param = run_parameter_study()
    figures.append(fig_param)
    fig_q10 = run_method_comparison(10.0)
    figures.append(fig_q10)
    fig_q01 = run_method_comparison(0.1)
    figures.append(fig_q01)
    fig_lorenz, fig_lorenz_sep = run_lorenz_sensitivity()
    figures.append(fig_lorenz)
    figures.append(fig_lorenz_sep)

    # ---- Exercise 2 figures ----
    fig_log_q01 = run_logistic_methods(0.1)
    figures.append(fig_log_q01)
    fig_log_q1 = run_logistic_methods(1.0)
    figures.append(fig_log_q1)
    fig_conv = run_convergence_study()
    figures.append(fig_conv)
    fig_forced, fig_diff = run_forced_lorenz()
    figures.append(fig_forced)
    figures.append(fig_diff)

    # Export individual figures
    savefig(fig_param, figs_dir / "param_sweep_cubic")
    savefig(fig_q10, figs_dir / "compare_q10")
    savefig(fig_q01, figs_dir / "compare_q01")
    savefig(fig_lorenz, figs_dir / "lorenz_sensitivity")
    savefig(fig_lorenz_sep, figs_dir / "lorenz_separation")
    savefig(fig_log_q01, figs_dir / "logistic_q0p1")
    savefig(fig_log_q1, figs_dir / "logistic_q1")
    savefig(fig_conv, figs_dir / "convergence_logistic")
    savefig(fig_forced, figs_dir / "forced_lorenz_midpoint_vs_euler")
    savefig(fig_diff, figs_dir / "forced_lorenz_difference")

    # Write canonical answers for exercise 1
    write_canonical_answers(run_dir / "answers.txt")

    # Create combined PDFs
    _create_results_pdf(run_dir, figures)

    # Close figures
    for fig in figures:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
