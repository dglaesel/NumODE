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
