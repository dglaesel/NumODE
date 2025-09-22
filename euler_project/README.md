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

The script produces refreshed plots in `euler_project/figs/` and overwrites
`euler_project/answers.txt` with a blank template for tasks (b)–(d) so you can
fill in your own notes after reviewing the figures.
