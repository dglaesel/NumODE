"""Plotting utilities for the Euler ODE experiments."""

from __future__ import annotations  # postpone evaluation of type hints (forward refs)

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection

mpl.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "lines.linewidth": 2.4,
    }
)
plt.style.use("ggplot")
DEFAULT_LW = 2.2

Array = np.ndarray


FIGS_DIR = Path("figs")


def ensure_dir(path: Path | str) -> Path:
    """Ensure that the directory exists and return its path object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def savefig(fig: Figure, filepath: Path | str) -> None:
    """Save a Matplotlib figure with sensible layout.

    - Applies tight_layout only for 2D figures (3D axes can misbehave).
    - Saves both PNG (300 dpi) and PDF (vector).
    """

    p = Path(filepath)
    has_3d = any(getattr(ax, "name", "") == "3d" for ax in fig.axes)
    if not has_3d:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")


def plot_param_sweep(t: Array, trajectories: Sequence[Array], qs: Sequence[float]) -> Figure:
    """Plot trajectories from the cubic parameter sweep, with clean layout."""

    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_pairs = sorted(zip(trajectories, qs), key=lambda z: z[1])
    for X, q in sorted_pairs:
        ax.plot(t, X[:, 0], label=f"q = {q}", linewidth=DEFAULT_LW)
        # subtle equilibrium guide (no legend entry)
        ax.axhline(np.sqrt(q), color="0.6", linestyle="--", linewidth=1.0, alpha=0.4, label="_nolegend_")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Cubic ODE parameter sweep (equilibria shown)")
    ax.grid(True)
    # Figure-level legend below the plot for clarity
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        title="Parameter",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.038),
        ncol=min(3, len(handles)),
        frameon=True,
    )
    fig.add_artist(leg)
    fig.subplots_adjust(right=0.95, left=0.12, top=0.9, bottom=0.22)
    return fig


def plot_compare_methods(
    t_euler_fine: Array,
    x_euler_fine: Array,
    t_euler_coarse: Array,
    x_euler_coarse: Array,
    t_ref: Array,
    x_ref: Array,
    q: float,
) -> Figure:
    """Create a comparison figure across numerical methods for a given q."""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_ref, x_ref[:, 0], label="LSODA", linewidth=DEFAULT_LW)
    ax.plot(
        t_euler_fine,
        x_euler_fine[:, 0],
        label=r"Euler $\tau$=0.01",
        linestyle="--",
        linewidth=DEFAULT_LW,
    )
    ax.plot(
        t_euler_coarse,
        x_euler_coarse[:, 0],
        label=r"Euler $\tau$=0.1",
        linestyle=":",
        linewidth=DEFAULT_LW,
    )
    y_eq = np.sqrt(q)
    ax.axhline(y_eq, linestyle="--", linewidth=1.2, alpha=0.6, color="0.5", label=r"$\sqrt{q}$")

    ef = np.interp(t_ref, t_euler_fine, x_euler_fine[:, 0])
    ec = np.interp(t_ref, t_euler_coarse, x_euler_coarse[:, 0])
    err_f = np.abs(ef - x_ref[:, 0])
    err_c = np.abs(ec - x_ref[:, 0])

    ax2 = ax.twinx()
    # Show error as soft bands to reduce visual competition with solution curves
    ax2.fill_between(t_ref, 0.0, err_f, color="tab:red", alpha=0.18, label=r"$|err|$ $\tau$=0.01")
    ax2.fill_between(t_ref, 0.0, err_c, color="tab:blue", alpha=0.18, label=r"$|err|$ $\tau$=0.1")
    ax2.set_ylabel("abs. error (Euler vs LSODA)")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(f"Method comparison for q = {q}")
    ax.grid(True)

    # Figure-level legends below the axes: separate boxes for Solution and Error
    sol_handles, sol_labels = ax.get_legend_handles_labels()
    err_handles, err_labels = ax2.get_legend_handles_labels()
    leg1 = fig.legend(
        sol_handles,
        sol_labels,
        title="Solution",
        loc="lower left",
        bbox_to_anchor=(0.02, -0.035),
        ncol=max(1, len(sol_handles)),
        frameon=True,
    )
    leg2 = fig.legend(
        err_handles,
        err_labels,
        title="Error",
        loc="lower right",
        bbox_to_anchor=(0.98, -0.035),
        ncol=max(1, len(err_handles)),
        frameon=True,
    )
    fig.add_artist(leg1)
    fig.add_artist(leg2)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.28)
    return fig


def plot_lorenz_with_timecolor(t: Array, X: Array, t2: Array, X2: Array) -> Figure:
    """Plot Lorenz trajectories with clear contrast – no inset."""

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Baseline: thin gray line only (avoid busy scatter)
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color="0.4", linewidth=1.2, alpha=0.7, label="Baseline")

    # Perturbed: highlight
    ax.plot(X2[:, 0], X2[:, 1], X2[:, 2], color="crimson", linewidth=2.6, label=r"$x_2(0)=5.01$ perturbed")
    ax.scatter3D(X[0, 0], X[0, 1], X[0, 2], color="black", s=18, marker="o")
    ax.scatter3D(X2[0, 0], X2[0, 1], X2[0, 2], color="crimson", s=26, marker="^")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("Lorenz sensitivity to perturbing $x_2(0)$")
    ax.view_init(elev=22, azim=-45)

    # Legend below the plot (figure-level)
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(handles),
        frameon=True,
    )
    fig.add_artist(leg)
    fig.subplots_adjust(bottom=0.22)

    return fig


def plot_lorenz_separation(t: Array, X: Array, X2: Array) -> Figure:
    """Plot the separation |ΔX| between two Lorenz trajectories over time."""

    sep = np.linalg.norm(X2 - X, axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, sep, color="crimson", linewidth=DEFAULT_LW)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"|$\Delta X$|(t)")
    ax.set_title("Lorenz trajectory separation")
    ax.grid(True)
    fig.subplots_adjust(bottom=0.14, left=0.12, right=0.95, top=0.88)
    return fig


__all__ = [
    "ensure_dir",
    "savefig",
    "plot_param_sweep",
    "plot_compare_methods",
    "plot_lorenz_with_timecolor",
    "plot_lorenz_separation",
]


def plot_logistic_comparison(
    t: Array,
    numeric_solutions: Sequence[Array],
    analytic: Array,
    method_labels: Sequence[str],
    q: float,
) -> Figure:
    """Plot several numerical solutions of the logistic ODE vs analytic.

    Parameters
    ----------
    t
        Time vector shared by all numerical solutions and the analytic values.
    numeric_solutions
        Sequence of arrays with shape ``(len(t), 1)``.
    analytic
        Array with shape ``(len(t),)`` or ``(len(t), 1)``.
    method_labels
        Legend labels for the numerical methods (same order as solutions).
    q
        Logistic parameter, used for the title.
    """

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    # Numeric methods: match the sample style — solid, dashed, dotted
    # Explicit colors and linestyles to make curves unmistakable
    palette = ["tab:blue", "tab:orange", "tab:green"]
    linestyles = ["-", "--", ":"]
    for idx, (X, lbl) in enumerate(zip(numeric_solutions, method_labels)):
        clr = palette[idx % len(palette)]
        ls = linestyles[idx % len(linestyles)]
        ax.plot(t, X[:, 0], label=lbl, linewidth=2.6, linestyle=ls, color=clr)

    # Analytic solution as open circle markers
    y = analytic.reshape(-1)
    ax.plot(
        t,
        y,
        linestyle="None",
        marker="o",
        markersize=5.0,
        markerfacecolor="none",
        markeredgecolor="tab:purple",
        label="analytic solution",
    )

    ax.set_xlabel("time")
    ax.set_ylabel("x(t)")
    ax.set_title("Approximation of the solution for different RK methods")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    # unobtrusive box identifying the q value
    ax.text(
        0.02,
        0.96,
        f"q = {q}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.4"},
    )
    fig.subplots_adjust(bottom=0.12, left=0.12, right=0.97, top=0.9)
    return fig


def plot_convergence(taus: Array, errors: dict[str, Array]) -> Figure:
    """Log–log convergence plot with fitted slopes.

    Plots ``log2(taus)`` vs ``log2(errors)``. Adds a linear fit per method and
    annotates the slope in the legend label.
    """

    taus = np.asarray(taus, dtype=float)
    if taus.ndim != 1:
        raise ValueError("taus must be a 1-D array")
    x = np.log2(taus)

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for i, (name, errs) in enumerate(errors.items()):
        e = np.asarray(errs, dtype=float).reshape(-1)
        # avoid log(0)
        eps = 1e-16
        e = np.maximum(e, eps)
        y = np.log2(e)
        # Fit y = m x + b
        m, b = np.polyfit(x, y, 1)
        label = f"{name} (slope≈{m:.2f})"
        clr = colors[i % len(colors)] if colors else None
        ax.plot(x, y, marker="o", markersize=6, linestyle="-", label=label, color=clr)
        # fitted line
        ax.plot(x, m * x + b, linestyle=":", color=clr, alpha=0.8)

    ax.set_xlabel(r"$\log_2(\tau)$")
    ax.set_ylabel(r"$\log_2(\mathrm{error})$")
    ax.set_title(r"$\log_2,\log_2$ plot of the error for the three different RK methods")
    ax.grid(True, which="both")
    ax.legend(loc="lower left", frameon=True)
    fig.subplots_adjust(bottom=0.14, left=0.12, right=0.97, top=0.9)
    return fig


def plot_lorenz_difference(t: Array, X_rk: Array, t_euler: Array, X_euler: Array) -> Figure:
    """Plot the norm of the difference between RK and explicit Euler vs time.

    Assumes both trajectories share the same uniform grid; if their lengths
    differ slightly, the plot uses the common prefix.
    """

    n = min(len(t), len(t_euler))
    tt = t[:n]
    diff = np.linalg.norm(X_rk[:n] - X_euler[:n], axis=1)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.semilogy(tt, diff, color="crimson", linewidth=2.4)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\|X_{\mathrm{RK}} - X_{\mathrm{Euler}}\|$")
    ax.set_title("Difference between midpoint RK and explicit Euler (forced Lorenz)")
    ax.grid(True, which="both")
    fig.subplots_adjust(bottom=0.14, left=0.12, right=0.97, top=0.9)
    return fig


def plot_forced_lorenz(
    t_rk: Array,
    X_rk: Array,
    t_euler: Array,
    X_euler: Array,
) -> Figure:
    """3‑D trajectories for the forced Lorenz system (midpoint vs Euler)."""

    fig = plt.figure(figsize=(9.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(X_rk[:, 0], X_rk[:, 1], X_rk[:, 2], color="tab:orange", label="midpoint RK")
    ax.plot(
        X_euler[:, 0],
        X_euler[:, 1],
        X_euler[:, 2],
        color="tab:blue",
        linestyle="--",
        label="explicit Euler",
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("Forced Lorenz trajectories")
    ax.view_init(elev=22, azim=-45)
    # Legend below to avoid covering the attractor
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles),
        frameon=True,
    )
    fig.add_artist(leg)
    fig.subplots_adjust(bottom=0.14, left=0.02, right=1.0, top=0.93)
    return fig


# Update exported names
__all__ += [
    "plot_logistic_comparison",
    "plot_convergence",
    "plot_forced_lorenz",
    "plot_lorenz_difference",
]


def plot_adaptive_solution(
    t_ad: Array,
    X_ad: Array,
    t_eu: Array,
    X_eu: Array,
    t_low: Array,
    X_low: Array,
    t_exact: Array,
    x_exact: Array,
    title: str = "Adaptive RK vs. references",
) -> Figure:
    """Overlay adaptive RK, explicit Euler, low-order RK and exact solution."""

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(t_exact, x_exact, color="black", label="exact", linewidth=2.6)
    ax.plot(t_eu, X_eu[:, 0], label="explicit Euler (tau=0.1)", linestyle=":")
    ax.plot(t_low, X_low[:, 0], label="Bogacki–Shampine low (tau=0.1)", linestyle="--")
    ax.plot(t_ad, X_ad[:, 0], label="adaptive embedded RK", color="tab:orange")
    ax.set_xlabel("time")
    ax.set_ylabel("x(t)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")
    fig.subplots_adjust(bottom=0.14, left=0.12, right=0.97, top=0.9)
    return fig


def plot_time_grids(grids: list[Array], labels: list[str], T: float, title: str = "Adaptive time grids") -> Figure:
    """Show discretization points for several time grids on one axis.

    Renders each grid as vertical tick marks on a baseline at different y-levels.
    """

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    y0 = 0.0
    dy = 1.0
    for i, (g, lab) in enumerate(zip(grids, labels)):
        y = y0 + i * dy
        ax.vlines(g, y - 0.35, y + 0.35, color=f"C{i}", alpha=0.9, label=lab)
    ax.set_xlim(0.0, float(T))
    ax.set_yticks([y0 + i * dy for i in range(len(grids))])
    ax.set_yticklabels(labels)
    ax.set_xlabel("time")
    ax.set_title(title)
    ax.grid(True, axis="x")
    fig.subplots_adjust(bottom=0.16, left=0.16, right=0.97, top=0.9)
    return fig


def plot_stepsizes_over_time(grids: list[Array], labels: list[str], title: str = "Step sizes over time") -> Figure:
    """Plot step sizes h_j = t_{j+1} - t_j against t_j for several grids."""

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for i, (g, lab) in enumerate(zip(grids, labels)):
        if len(g) >= 2:
            h = np.diff(g)
            ax.step(g[:-1], h, where="post", label=lab)
    ax.set_xlabel("time")
    ax.set_ylabel("step size")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", ncol=2)
    fig.subplots_adjust(bottom=0.14, left=0.12, right=0.97, top=0.9)
    return fig


def plot_3d_single(t: Array, X: Array, title: str = "Adaptive RK trajectory (Lorenz)") -> Figure:
    """3D trajectory without comparison curve."""

    fig = plt.figure(figsize=(9.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color="tab:orange", label="embedded RK")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-45)
    fig.subplots_adjust(bottom=0.08, left=0.02, right=1.0, top=0.93)
    return fig


__all__ += [
    "plot_adaptive_solution",
    "plot_time_grids",
    "plot_stepsizes_over_time",
    "plot_3d_single",
]


def plot_multi_approximations(
    curves: list[tuple[Array, Array, str]],
    t_exact: Array,
    x_exact: Array,
    title: str = "Adaptive solutions for various TOL",
) -> Figure:
    """Overlay multiple solution curves with exact solution.

    Parameters
    - curves: list of tuples (t, X, label) where X is shaped (n, d); uses first component.
    - t_exact, x_exact: reference solution to overlay.
    """

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(t_exact, x_exact, color="black", linewidth=2.6, label="exact")
    for i, (t, X, lab) in enumerate(curves):
        ax.plot(t, X[:, 0], label=lab)
    ax.set_xlabel("time")
    ax.set_ylabel("x(t)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", ncol=2)
    fig.subplots_adjust(bottom=0.14, left=0.12, right=0.97, top=0.9)
    return fig


__all__ += ["plot_multi_approximations"]


def plot_error_curves(
    err_curves: list[tuple[Array, Array, str]],
    title: str = "Grid error over time",
) -> Figure:
    """Plot absolute error vs time for several methods/tolerances.

    err_curves: list of (t, err, label) tuples.
    """

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for t, e, lab in err_curves:
        ax.semilogy(t, e, label=lab)
    ax.set_xlabel("time")
    ax.set_ylabel("abs. error")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend(loc="best", ncol=2)
    fig.subplots_adjust(bottom=0.14, left=0.12, right=0.97, top=0.9)
    return fig


def plot_runtime_vs_error(
    tols: Array,
    runtimes: Array,
    errors: Array,
    steps: Array | None = None,
    title: str = "Accuracy vs. runtime (adaptive RK)",
) -> Figure:
    """Scatter of runtime (x) against error (y) for different tolerances.

    - ``errors`` can be the max or final-time error.
    - Points are annotated with the tolerance value; if ``steps`` is given,
      it's included in the annotation.
    """

    tols = np.asarray(tols, dtype=float)
    runtimes = np.asarray(runtimes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.loglog(runtimes, errors, marker="o", linestyle="-", color="tab:blue")
    for i, (rt, er) in enumerate(zip(runtimes, errors)):
        label = f"TOL={tols[i]:.0e}"
        if steps is not None:
            label += f"\nN={int(steps[i])}"
        ax.annotate(label, (rt, er), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("runtime [s]")
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.grid(True, which="both")
    fig.subplots_adjust(bottom=0.14, left=0.15, right=0.97, top=0.9)
    return fig


__all__ += ["plot_error_curves", "plot_runtime_vs_error"]


def plot_tradeoff_table(
    tols: Array,
    runtimes: Array,
    steps: Array,
    max_errors: Array,
    final_errors: Array,
    title: str = "TOL trade-off metrics (adaptive RK)",
) -> Figure:
    """Render a compact table summarizing accuracy vs efficiency metrics.

    Columns: TOL, runtime [s], steps, max error, final error.
    """

    tols = np.asarray(tols).reshape(-1)
    runtimes = np.asarray(runtimes).reshape(-1)
    steps = np.asarray(steps).reshape(-1)
    max_errors = np.asarray(max_errors).reshape(-1)
    final_errors = np.asarray(final_errors).reshape(-1)

    cell_text: list[list[str]] = []
    for i in range(len(tols)):
        row = [
            f"{tols[i]:.0e}",
            f"{runtimes[i]:.6f}",
            f"{int(steps[i])}",
            f"{max_errors[i]:.3e}",
            f"{final_errors[i]:.3e}",
        ]
        cell_text.append(row)

    columns = ["TOL", "runtime [s]", "N steps", "max error", "final error"]

    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.axis("off")
    ax.set_title(title, pad=12)
    tbl = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.1, 1.3)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.15)
    return fig


__all__ += ["plot_tradeoff_table"]
