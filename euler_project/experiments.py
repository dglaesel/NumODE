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
from .experiment3 import run_arctan_problem, run_tol_influence, run_lorenz_adaptive
from .experiment4 import (
    run_fixed_point_families,
    run_lorenz_implicit,
    run_lorenz_forced_const,
    run_lorenz_forced_sin,
    make_lorenz_fixedpoint_table,
)
from .experiment5 import (
    run_stability_cubic,
    run_oscillator_phase,
    run_order_study,
)


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

    # ---- Exercise 3 figures (adaptive embedded RK) ----
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
    if "fig_table" in metrics:
        figures.append(metrics["fig_table"])  # type: ignore[index]
    fig_d_3d, fig_d_h = run_lorenz_adaptive()
    figures.append(fig_d_3d)
    figures.append(fig_d_h)

    # ---- Exercise 4 figures (implicit Euler) ----
    figs_c = run_fixed_point_families(q=9.0, T=2.0, tau=0.05)
    for fig in figs_c:
        figures.append(fig)
    fig_d_imp = run_lorenz_implicit()
    fig_d_const = run_lorenz_forced_const()
    fig_d_sin = run_lorenz_forced_sin()
    fig_d_table = make_lorenz_fixedpoint_table()
    figures.append(fig_d_imp)
    figures.append(fig_d_const)
    figures.append(fig_d_sin)
    figures.append(fig_d_table)

    # ---- Exercise 5 figures (implicit RK and isometries) ----
    fig_e2_all, fig_e2_zoom = run_stability_cubic()
    figures.append(fig_e2_all)
    figures.append(fig_e2_zoom)
    fig_e3_undamped = run_oscillator_phase(alpha=0.0)
    fig_e3_damped = run_oscillator_phase(alpha=0.02)
    figures.append(fig_e3_undamped)
    figures.append(fig_e3_damped)
    fig_e4_order = run_order_study()
    figures.append(fig_e4_order)

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
    # Exercise 3
    savefig(fig_b_sol, figs_dir / "ex3_b_adaptive_vs_refs")
    savefig(fig_b_grid, figs_dir / "ex3_b_adaptive_grid")
    savefig(fig_b_steps, figs_dir / "ex3_b_adaptive_stepsize")
    savefig(fig_c_sol, figs_dir / "ex3_c_solutions_vs_tol")
    savefig(fig_c_grid, figs_dir / "ex3_c_grids_vs_tol")
    savefig(fig_c_h, figs_dir / "ex3_c_stepsizes_vs_tol")
    savefig(fig_c_err, figs_dir / "ex3_c_grid_errors_vs_tol")
    savefig(fig_trade, figs_dir / "ex3_c_accuracy_vs_runtime")
    if "fig_table" in metrics:
        savefig(metrics["fig_table"], figs_dir / "ex3_c_tradeoff_table")  # type: ignore[index]
    savefig(fig_d_3d, figs_dir / "ex3_d_lorenz_adaptive_3d")
    savefig(fig_d_h, figs_dir / "ex3_d_lorenz_stepsizes")

    # Exercise 4
    names_c = [
        "ex4_c_fixed_x0_explicit",
        "ex4_c_fixed_x0_implicit",
        "ex4_c_fixed_xminus_explicit",
        "ex4_c_fixed_xminus_implicit",
        "ex4_c_fixed_xplus_explicit",
        "ex4_c_fixed_xplus_implicit",
    ]
    for fig, name in zip(figs_c, names_c):
        savefig(fig, figs_dir / name)
    savefig(fig_d_imp, figs_dir / "ex4_d_lorenz_implicit_3d")
    savefig(fig_d_const, figs_dir / "ex4_d_lorenz_forcing_const_3d")
    savefig(fig_d_sin, figs_dir / "ex4_d_lorenz_forcing_sin_3d")
    savefig(fig_d_table, figs_dir / "ex4_d_lorenz_fixedpoint_table")

    # Exercise 5
    savefig(fig_e2_all, figs_dir / "ex5_task2_approximations_1d")
    savefig(fig_e2_zoom, figs_dir / "ex5_task2_approximations_1d_zoom")
    savefig(fig_e3_undamped, figs_dir / "ex5_task3_phase_undamped")
    savefig(fig_e3_damped, figs_dir / "ex5_task3_phase_damped")
    savefig(fig_e4_order, figs_dir / "ex5_task4_convergence_order")

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
