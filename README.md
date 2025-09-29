# NumODE — Euler and Runge–Kutta for ODEs

Assignments 1 and 2 in one repo. It started as an Explicit Euler baseline and now includes a generic explicit Runge–Kutta solver with additional experiments from programming exercise 3.4. The driver creates per-run, timestamped results that are easy to review and share.

## Setup (Windows, PowerShell)
1. Clone and enter the repo
   ```powershell
   git clone https://github.com/dglaesel/NumODE.git
   cd NumODE
   ```
2. Create a virtual environment (use your installed Python 3.x)
   ```powershell
   py -3 -m venv .venv
   ```
3. Activate the environment (fix policy once if needed)
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies
   ```powershell
   pip install -r requirements.txt
   ```
5. Run all experiments (assignments 1 + 2)
   ```powershell
   python -m euler_project.experiments
   ```

### Run a single assignment
If you only want to run one assignment's figures:

- Exercise 1 (original):
  ```powershell
  python -m euler_project.experiment1
  ```
- Exercise 2 (Runge–Kutta):
  ```powershell
  python -m euler_project.experiment2
  ```

## Output
Every run creates a folder: `euler_project/runs/<YYYYMMDD-HHMMSS>/`
- `figs/` - PNG and PDF for each plot
- `all_plots.pdf` - concatenation of the raw Matplotlib figure pages
- `answers.txt` - template for Exercise 1 answers (optional)
- `results.pdf` - plots first, then the content of `answers.txt` (if present)

## Project Architecture
- `euler_project/integrators.py`
  - `ExplicitEuler` + `explEuler(f, x0, T, tau)`.
  - `ExplicitRungeKutta` + `exRungeKutta(f, x0, T, tau, A, b, c)` (generic ERK driven by a Butcher tableau; supports autonomous and non-autonomous RHS and vector states).
- `euler_project/problems.py`
  - `rhs_cubic`, `rhs_lorenz` (exercise 1).
  - `rhs_logistic`, closed form `logistic_analytic`, and `rhs_forced_lorenz` (exercise 2).
- `euler_project/plotting.py`
  - Utilities (`ensure_dir`, `savefig`) and figure producers:
    - exercise 1: parameter sweep, method comparison, Lorenz sensitivity;
    - exercise 2: logistic comparison, convergence plot, forced Lorenz 3-D.
- `euler_project/experiment1.py`
  - Assignment 1 experiments (parameter sweep, LSODA comparison, Lorenz sensitivity).
- `euler_project/experiment2.py`
  - Assignment 2 experiments (logistic method comparison, convergence study, forced Lorenz with midpoint RK vs Euler). Runnable as a standalone module.
- `euler_project/experiments.py`
  - Aggregator CLI to run everything and build per-run PDFs.

### Meta/Extensibility Guidelines
- New integrators
  - Add them to `euler_project/integrators.py`. Keep the public API consistent (class with `.run()` returning `(t, X)` and a small function wrapper).
- New problems (future sheets)
  - Implement additional right-hand sides in `euler_project/problems.py`.
  - Prefer pure functions with NumPy arrays; document parameters and shapes.
- New experiments
  - Add a new `euler_project/experimentN.py` with `run_*` helpers and a `main()` that saves figures and builds PDFs. Also import it from `euler_project/experiments.py` to include in the “run everything” mode.
- Per-run bookkeeping
  - The drivers write to `euler_project/runs/<timestamp>/` so multiple runs never overwrite each other and can be shared/reviewed easily.

## Notes
- Tested with Python 3.13, should work with 3.10+.
- Useful commands:
  - List installed Pythons: `py --list`
  - Deactivate venv: `deactivate`
