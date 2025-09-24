# NumODE — Explicit Euler for ODEs

Assignment 1/Sheet 2.3 style project for experimenting with numerical
integration of ordinary differential equations (ODEs). The repo provides a
clean Explicit Euler baseline, a small set of problems, plotting helpers, and
an experiment driver that produces per‑run, timestamped results.

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
5. Run the experiments
   ```powershell
   python -m euler_project.experiments
   ```

## Output
Every run creates a folder: `euler_project/runs/<YYYYMMDD-HHMMSS>/`
- `figs/` — PNG and PDF for each plot
- `all_plots.pdf` — concatenation of the raw Matplotlib figure pages
- `answers.txt` — blank template you can fill manually
- `results.pdf` — plots first, then the content of `answers.txt`

## Project Architecture
- `euler_project/integrators.py`
  - Explicit Euler integrator. Exposes the `ExplicitEuler` class and the
    convenience function `explEuler(f, x0, T, tau)` used across experiments.
  - Extensible: this module is the right place to implement additional
    time‑stepping methods (e.g., Implicit Euler, Heun, RK4, adaptive schemes).
- `euler_project/problems.py`
  - Right‑hand sides for the ODEs used in the assignments: `rhs_cubic`
    (scalar cubic IVP) and `rhs_lorenz` (Lorenz–63 system).
  - Extensible: add new problem functions here as future sheets introduce new
    models. Keep signatures tidy and NumPy‑friendly.
- `euler_project/plotting.py`
  - Plotting utilities: `ensure_dir`, `savefig`, and the figure producers for
    each experiment (parameter sweep, method comparison, Lorenz).
- `euler_project/experiments.py`
  - The orchestration layer/CLI entry point. It
    - runs (b) parameter sweep (`run_parameter_study`),
    - runs (c) method comparisons for `q = 10` and `q = 0.1`
      (`run_method_comparison`),
    - runs (d) Lorenz sensitivity (`run_lorenz_sensitivity`),
    - saves individual figures to `figs/`,
    - writes `answers.txt`, and
    - builds `results.pdf` (plots then answers).

### Meta/Extensibility Guidelines
- New integrators
  - Add them to `euler_project/integrators.py` alongside Explicit Euler. Keep
    the public API consistent (e.g., `implEuler(f, x0, T, tau)` or a class with
    `.run()` that returns `(t, X)`).
- New problems (future sheets)
  - Implement additional right‑hand sides in `euler_project/problems.py`.
  - Prefer pure functions with NumPy arrays; document parameters and shapes.
- New experiments
  - Add a `run_*()` helper in `euler_project/experiments.py` that produces a
    `matplotlib.figure.Figure` (and saves it via `savefig`).
  - Compose figures into `results.pdf` by editing `_create_results_pdf()` order
    or appending a new page.
- Per‑run bookkeeping
  - The driver always writes to `euler_project/runs/<timestamp>/` so multiple
    runs never overwrite each other and can be shared/reviewed easily.

## Notes
- Tested with Python 3.13, should work with 3.10+.
- Useful commands:
  - List installed Pythons: `py --list`
  - Deactivate venv: `deactivate`
