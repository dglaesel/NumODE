# Euler/Runge–Kutta Project

This package now covers programming exercises 1 and 2. It started with an
explicit Euler baseline and has been extended with a generic explicit
Runge–Kutta (ERK) solver to run the new experiments from exercise 3.4.

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

Run one assignment only

- Exercise 1 (original): `python -m euler_project.experiment1`
- Exercise 2 (Runge–Kutta): `python -m euler_project.experiment2`

Each run creates a timestamped output folder under `euler_project/runs/`:

- Root: `euler_project/runs/<YYYYMMDD-HHMMSS>/`
- Figures: `<run>/figs/` (PNG + PDF)
- Answers: `<run>/answers.txt` (exercise 1 write‑up)
- Combined PDFs: `<run>/all_plots.pdf` and `<run>/results.pdf`

## Architecture

- `integrators.py`
  - Original: `ExplicitEuler` + `explEuler(f, x0, T, tau)`.
  - New: `ExplicitRungeKutta` + `exRungeKutta(f, x0, T, tau, A, b, c)` – a
    generic ERK solver for any explicit Butcher tableau; works for autonomous
    and non‑autonomous RHS and vector states.
- `problems.py`
  - Original: `rhs_cubic`, `rhs_lorenz`.
  - New: `rhs_logistic`, its closed form `logistic_analytic`, and
    `rhs_forced_lorenz` (sinusoidal forcing in the first equation).
- `plotting.py`
  - Original plotting utilities preserved.
  - New: `plot_logistic_comparison`, `plot_convergence`, `plot_forced_lorenz`.
- `experiment1.py`
  - The previous experiment suite (exercise 1) kept intact.
- `experiment2.py`
  - New RK experiments (exercise 2): logistic method comparison, convergence
    study, and forced Lorenz.
- `experiments.py`
  - Aggregator entry point that runs both experiment sets and exports results.
  - To run only exercise 1 or 2 directly:
    - `python -m euler_project.experiment1`
    - `python -m euler_project.experiment2`

### Data Flow
- Numerical steps come from `integrators.explEuler` and the new
  `integrators.exRungeKutta`.
- Arrays feed the plotters in `plotting.py` to create figures.
- `experiments.py` handles export and result collation per run.
