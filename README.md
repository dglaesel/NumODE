# NumODE – Explicit Euler for ODEs
Assignment 1

## 📦 Setup (Windows, PowerShell)
1. **Clone the repository**
   ```powershell
   git clone https://github.com/dglaesel/NumODE.git
   cd NumODE
   ```
2. **Create a virtual environment**  
   Use the latest installed Python 3.x (check with `py --list`):
   ```powershell
   py -3 -m venv .venv
   ```
3. **Activate the environment**  
   If you get a security error about `Activate.ps1`, run once:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```
   Then activate:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
5. **Run experiments**
   ```powershell
   python -m euler_project.experiments
   ```

## 📂 Output
- Figures are saved in: `euler_project/figs/`  
- Answer template is generated/updated in: `euler_project/answers.txt`

## 🔧 Notes
- Tested with **Python 3.13**, but should work with any Python ≥3.10.  
- To see your installed Python versions: `py --list`  
- To deactivate the virtual environment: `deactivate
