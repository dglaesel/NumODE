# NumODE

This repository implements assignment 2.3 (Sheet 2) — Explicit Euler method.

## Setup

Create and activate a virtual environment (for example, in the repository root)
before installing the dependencies.

### Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the experiments

Run the orchestrating module from an activated environment:

```bash
python -m euler_project.experiments
```

This command refreshes the required PNG figures in `euler_project/figs/` and
rewrites `euler_project/answers.txt` with a blank template for tasks (b)–(d)
so you can record your own observations.
