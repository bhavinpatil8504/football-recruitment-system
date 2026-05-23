"""
setup.py — Cross-platform environment setup
--------------------------------------------
Fallback if setup.bat / setup.sh don't work.
Run from inside the project folder:

    python setup.py

This script installs packages in the correct dependency order
into the CURRENT active environment (or venv if you activated one).
"""

import subprocess
import sys
import os


def run(cmd: list, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n  ERROR during: {label}")
        print("  Try running this command manually and check the error:")
        print("  " + " ".join(cmd))
        sys.exit(1)


def verify(package: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {package}; "
             f"print('{package}', getattr({package}, '__version__', 'OK'))"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✓  {result.stdout.strip()}")
            return True
        else:
            print(f"  ✗  {package} — {result.stderr.strip()[:80]}")
            return False
    except Exception as e:
        print(f"  ✗  {package}: {e}")
        return False


pip = [sys.executable, "-m", "pip"]

print("="*60)
print("  Football Recruitment — Environment Setup")
print(f"  Python: {sys.version}")
print("="*60)

# ── Step 1: Upgrade pip ───────────────────────────────────────────────────────
run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"],
    "Upgrading pip, setuptools, wheel")

# ── Step 2: Foundation — MUST come before everything else ────────────────────
run(pip + ["install",
           "numpy==1.26.4",
           "pandas==2.1.4",
           "scipy==1.11.4",
           "pyarrow>=14.0.0"],
    "Step 2/9 — Foundation: numpy, pandas, scipy, pyarrow")

# ── Step 3: ML core ──────────────────────────────────────────────────────────
run(pip + ["install",
           "scikit-learn==1.3.2",
           "xgboost==2.0.3"],
    "Step 3/9 — ML: scikit-learn, xgboost")

# ── Step 4: Football data tools ───────────────────────────────────────────────
run(pip + ["install",
           "statsbombpy>=1.18.0",
           "socceraction==1.4.2"],
    "Step 4/9 — Data: statsbombpy (1.18+), socceraction")

# ── Step 5: Bayesian modeling — REMOVED ──────────────────────────────────────
# PyMC replaced with pure numpy Empirical Bayes (no C compiler needed).
print("\n  Step 5/9 — Bayesian: using built-in numpy implementation (no install needed)")

# ── Step 6: Optimization ─────────────────────────────────────────────────────
run(pip + ["install", "pulp==2.7.0"],
    "Step 6/9 — Optimization: pulp")

# ── Step 7: Visualization + utilities ────────────────────────────────────────
run(pip + ["install",
           "matplotlib==3.8.2",
           "seaborn==0.13.2",
           "plotly==5.18.0",
           "tqdm==4.66.1",
           "joblib==1.3.2",
           "shap"],
    "Step 7/9 — Visualization + utilities + SHAP")

# ── Step 8: PyTorch (CPU) ─────────────────────────────────────────────────────
run(pip + ["install",
           "torch==2.1.2",
           "torchvision==0.16.2",
           "--index-url", "https://download.pytorch.org/whl/cpu"],
    "Step 8/9 — PyTorch (CPU)")

# ── Step 9: PyTorch Geometric ─────────────────────────────────────────────────
run(pip + ["install", "torch-geometric==2.4.0"],
    "Step 9/9 — PyTorch Geometric (GNN)")

# Try optional PyG sparse kernels (may not be available on all platforms)
print("\n  Installing optional PyG sparse kernels (may be skipped)...")
result = subprocess.run(
    pip + ["install",
           "pyg-lib", "torch-scatter", "torch-sparse",
           "torch-cluster", "torch-spline-conv",
           "-f", "https://data.pyg.org/whl/torch-2.1.0+cpu.html"],
    capture_output=True
)
if result.returncode == 0:
    print("  ✓  PyG sparse kernels installed")
else:
    print("  ⚠  PyG sparse kernels not available for this platform "
          "(torch-geometric still works without them)")

# ── Verification ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Verifying installation")
print("="*60)

packages = [
    "numpy", "pandas", "scipy", "sklearn", "xgboost",
    "statsbombpy", "socceraction", "pulp",
    "torch", "torch_geometric",
]
results = [verify(p) for p in packages]
failed  = [p for p, ok in zip(packages, results) if not ok]

print("\n" + "="*60)
if failed:
    print(f"  ⚠  Some packages failed: {failed}")
    print("  Try installing them manually:")
    for p in failed:
        print(f"    pip install {p}")
else:
    print("  ✓  All packages installed successfully")
    print()
    print("  Next steps:")
    print("    python pipeline.py --quick --skip-gnn   # fast test (~5 min)")
    print("    python pipeline.py --n-seasons 3        # full run (~45 min)")
print("="*60)
