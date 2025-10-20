# NumODE – Explicit/Embedded Runge–Kutta for ODEs

Programming exercises 1–3 in one repo. It started with an Explicit Euler
baseline and now includes a generic explicit Runge–Kutta integrator and an
adaptive embedded RK solver (Bogacki–Shampine 3(2)) with step‑size control.
Drivers create per‑run, timestamped results that are easy to review and share.

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
5. Run all experiments (exercises 1 + 2 + 3)
   ```powershell
   python -m euler_project.experiments
   ```

### Run a single experiment
If you only want to run one exercise’s figures:

- Exercise 1 (original):
  ```powershell
  python -m euler_project.experiment1
  ```
- Exercise 2 (Runge–Kutta):
  ```powershell
  python -m euler_project.experiment2
  ```
- Exercise 3 (adaptive embedded RK):
  ```powershell
  python -m euler_project.experiment3
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
  - `ExplicitRungeKutta` + `exRungeKutta(f, x0, T, tau, A, b, c)` (generic ERK driven by a Butcher tableau; supports autonomous and non‑autonomous RHS and vector states).
  - `EmbeddedRungeKuttaAdaptive` + `adaptive_embedded_rk(f, x0, T, tauMax, rho, q, TOL, A, b_high, b_low, c, p_error)` implementing adaptive step‑size control using an embedded pair (used here with Bogacki–Shampine 3(2)). Returns `(t_grid, X_sol)`.
- `euler_project/problems.py`
  - `rhs_cubic`, `rhs_lorenz` (exercise 1).
  - `rhs_logistic`, closed form `logistic_analytic`, and `rhs_forced_lorenz` (exercise 2).
  - `rhs_cos2_arctan_problem` and `arctan_analytic` (exercise 3b/c).
- `euler_project/plotting.py`
  - Utilities (`ensure_dir`, `savefig`) and figure producers for all three exercises (comparison plots, adaptive grids, step sizes, 3‑D Lorenz).
- `euler_project/experiment1.py`
  - Exercise 1 experiments (parameter sweep, LSODA comparison, Lorenz sensitivity).
- `euler_project/experiment2.py`
  - Exercise 2 experiments (logistic method comparison, convergence study, forced Lorenz with midpoint RK vs Euler). Runnable as a standalone module.
- `euler_project/experiment3.py`
  - Exercise 3 (adaptive):
    - b) Arctan problem – compare adaptive BS23 vs Euler, BS(2) and exact; plus plot of adaptive time grid.
    - c) TOL study – show approximations for TOL = 1e-3..1e-8, the chosen grids and the step sizes over time.
    - d) Forced Lorenz – 3‑D trajectory (x1,x2,x3) and step sizes.
- `euler_project/experiments.py`
  - Aggregator CLI to run everything (ex1+ex2+ex3) and build per‑run PDFs.

### Meta/Extensibility Guidelines
- New integrators
  - Add them to `euler_project/integrators.py`. Keep the public API consistent (class with `.run()` returning `(t, X)` and a small function wrapper).
- New problems (future sheets)
  - Implement additional right-hand sides in `euler_project/problems.py`.
  - Prefer pure functions with NumPy arrays; document parameters and shapes.
- New experiments
  - Add a new `euler_project/experimentN.py` with `run_*` helpers and a `main()` that saves figures and builds PDFs. Also import it from `euler_project/experiments.py` to include in the "run everything" mode.
- Per-run bookkeeping
  - The drivers write to `euler_project/runs/<timestamp>/` so multiple runs never overwrite each other and can be shared/reviewed easily.

## Notes
- Tested with Python 3.13, should work with 3.10+.
- Useful commands:
  - List installed Pythons: `py --list`
  - Deactivate venv: `deactivate`

