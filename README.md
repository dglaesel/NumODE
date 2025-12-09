# NumODE - Explicit and Implicit Runge-Kutta for ODEs

Executive Summary: This repo is a teaching lab for solving ordinary differential
equations in Python. It implements a small collection of time-stepping methods
(explicit Euler, explicit and adaptive Runge-Kutta, implicit Euler, implicit
Runge-Kutta) and applies them to textbook problems (logistic growth, Lorenz
system, oscillators). Running the provided drivers generates plots and PDFs that
compare methods, study stability/accuracy, and show adaptive step-size behavior.

Programming exercises 1-5 in one place. The code now covers explicit Euler,
explicit Runge-Kutta, adaptive embedded RK (Bogacki-Shampine 3(2)),
implicit Euler (general and linear) and a generic implicit Runge-Kutta solver.
Drivers create per-run, timestamped results that are easy to review and share.

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
5. Run all experiments (exercises 1-5)
   ```powershell
   python -m euler_project.experiments
   ```

### Run a single experiment
If you only want to run one exercise:

- Exercise 1 (original):
  ```powershell
  python -m euler_project.experiment1
  ```
- Exercise 2 (Runge-Kutta):
  ```powershell
  python -m euler_project.experiment2
  ```
- Exercise 3 (adaptive embedded RK):
  ```powershell
  python -m euler_project.experiment3
  ```
- Exercise 4 (implicit Euler):
  ```powershell
  python -m euler_project.experiment4
  ```
- Exercise 5 (implicit Runge-Kutta and oscillators):
  ```powershell
  python -m euler_project.experiment5
  ```

## Output
Every run creates a folder under `euler_project/runs/` (the standalone drivers
prefix the folder with `exN-<timestamp>`):
- `figs/` - PNG and PDF for each plot.
- `all_plots.pdf` - concatenation of the raw Matplotlib figure pages.
- `answers.txt` - filled when running exercise 1 (aggregator only).
- `results.pdf` - plots first, then `answers.txt` appended for exercise 1 runs.

## Project Architecture
- `euler_project/integrators.py`
  - `ExplicitEuler` + `explEuler(f, x0, T, tau)`.
  - `ExplicitRungeKutta` + `exRungeKutta(f, x0, T, tau, A, b, c)` (generic ERK driven by a Butcher tableau; supports autonomous and non-autonomous RHS and vector states).
  - `EmbeddedRungeKuttaAdaptive` + `adaptive_embedded_rk(...)` (adaptive step-size control using an embedded pair; used with Bogacki-Shampine 3(2)).
  - `ImplicitEuler` + `implicitEuler(...)` (nonlinear step solve) and `ImplicitEulerLinear` + `implicitEuler_linear(...)` for affine systems.
  - `ImplicitRungeKutta` + `implicitRungeKutta(...)` and alias `implicitRK(...)` (generic IRK with configurable tableau and solver).
- `euler_project/problems.py`
  - `rhs_cubic`, `rhs_lorenz` (exercise 1).
  - `rhs_logistic`, closed form `logistic_analytic`, and `rhs_forced_lorenz` (exercise 2).
  - `rhs_cos2_arctan_problem` and `arctan_analytic` (exercise 3b/c).
  - `rhs_harmonic_oscillator` and `oscillator_exact_undamped` (exercise 5).
- `euler_project/plotting.py`
  - Utilities (`ensure_dir`, `savefig`) and figure producers across all exercises (comparison plots, adaptive grids, step sizes, 3-D Lorenz, tables, phase plots).
- `euler_project/experiment1.py`
  - Exercise 1 experiments (parameter sweep, LSODA comparison, Lorenz sensitivity).
- `euler_project/experiment2.py`
  - Exercise 2 experiments (logistic method comparison, convergence study, forced Lorenz with midpoint RK vs Euler). Runnable as a standalone module.
- `euler_project/experiment3.py`
  - Exercise 3 (adaptive):
    - b) Arctan problem - compare adaptive BS23 vs Euler, BS(2) and exact; plus plot of adaptive time grid.
    - c) TOL study - show approximations for TOL = 1e-3..1e-8, the chosen grids and the step sizes over time.
    - d) Forced Lorenz - 3-D trajectory (x1,x2,x3) and step sizes.
- `euler_project/experiment4.py`
  - Exercise 4 (implicit Euler): cubic fixed-point families (explicit vs implicit overlays) and Lorenz runs (unforced, constant forcing, sinusoidal forcing) plus a fixed-point distance table.
- `euler_project/experiment5.py`
  - Exercise 5 (implicit Runge-Kutta and oscillators): stability of the cubic ODE with four methods, oscillator phase portraits (undamped/damped), and a convergence-order study.
- `euler_project/experiments.py`
  - Aggregator CLI to run everything (ex1-ex5) and build per-run PDFs with answers appended for exercise 1.

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

