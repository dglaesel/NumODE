# Euler Project

This package solves assignment 2.3 from Sheet 2 using an explicit Euler
integrator for ordinary differential equations.

## Running the experiments

Activate your virtual environment and install dependencies from the repository
root before executing the experiment driver.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m euler_project.experiments
```

Each run now creates a timestamped output folder under `euler_project/runs/`:

- Root: `euler_project/runs/<YYYYMMDD-HHMMSS>/`
- Figures: `<run>/figs/` (PNG files)
- Answers: `<run>/answers.txt` (blank template to fill)

Note: Older instructions that referenced `euler_project/figs/` and
`euler_project/answers.txt` are superseded by the per-run layout above.

## Architecture

- `integrators.py`
  - Provides the explicit Euler method via the `ExplicitEuler` class and the
    convenience function `explEuler(f, x0, T, tau)` used by the experiments.
- `problems.py`
  - Right‑hand side functions for the assignment: `rhs_cubic` (scalar cubic
    ODE) and `rhs_lorenz` (Lorenz–63 system).
- `plotting.py`
  - Small utilities (`ensure_dir`, `savefig`) and purpose‑built plotting
    functions for each experiment: parameter sweep, method comparison, and
    Lorenz visualisations (3D trajectory and a separate separation plot).
- `experiments.py`
  - The driver you run with `python -m euler_project.experiments`. It:
    - Runs (b) parameter sweep (`run_parameter_study`).
    - Runs (c) method comparisons for `q=10` and `q=0.1`
      (`run_method_comparison`).
    - Runs (d) Lorenz sensitivity (`run_lorenz_sensitivity`).
    - Creates a timestamped run folder with exported figures (`figs/`), a
      blank `answers.txt`, `all_plots.pdf` (raw Matplotlib pages), and a
      user‑facing `results.pdf` that appends your answers at the end.

### Data Flow
- Numerical steps come from `integrators.explEuler`.
- Arrays feed the plotters in `plotting.py` to create figures.
- `experiments.py` handles export and result collation per run.
