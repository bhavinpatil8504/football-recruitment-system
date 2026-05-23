#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  setup.sh  —  Mac / Linux setup for football_recruitment
#  Run once from inside the project folder:
#    chmod +x setup.sh && ./setup.sh
#
#  Prerequisites: Python 3.10 or 3.11
#  Check with:    python3 --version
# ─────────────────────────────────────────────────────────────────────────────

set -e   # stop on first error

echo "Checking Python version..."
python3 --version

echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv

echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Step 3: Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 4: Installing numpy + pandas + scipy FIRST (foundation layer)..."
pip install "numpy==1.26.4" "pandas==2.1.4" "scipy==1.11.4"

echo ""
echo "Step 5: Installing scikit-learn and xgboost..."
pip install "scikit-learn==1.3.2" "xgboost==2.0.3"

echo ""
echo "Step 6: Installing StatsBomb data tools..."
pip install "statsbombpy==1.0.3" "socceraction==1.4.2"

echo ""
echo "Step 7: Bayesian — using built-in numpy implementation (no install needed)"

echo ""
echo "Step 8: Installing PuLP (optimization)..."
pip install "pulp==2.7.0"

echo ""
echo "Step 9: Installing visualization and utilities..."
pip install "matplotlib==3.8.2" "seaborn==0.13.2" "plotly==5.18.0" \
            "tqdm==4.66.1" "joblib==1.3.2" "shap"

echo ""
echo "Step 10: Installing PyTorch (CPU version)..."
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Step 11: Installing PyTorch Geometric (GNN)..."
pip install torch-geometric==2.4.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html 2>/dev/null || \
    echo "  PyG extras not available for this platform — torch-geometric still works."

echo ""
echo "Step 12: Verifying installation..."
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import pandas; print('pandas', pandas.__version__)"
python3 -c "import scipy; print('scipy', scipy.__version__)"
python3 -c "import sklearn; print('sklearn', sklearn.__version__)"
python3 -c "import xgboost; print('xgboost', xgboost.__version__)"
python3 -c "import statsbombpy; print('statsbombpy OK')"
python3 -c "import socceraction; print('socceraction OK')"
python3 -c "import shap; print('shap', shap.__version__)"
python3 -c "import pulp; print('pulp', pulp.__version__)"
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"

echo ""
echo "─────────────────────────────────────────────────────────────────────────"
echo " Setup complete."
echo " To activate this environment in future sessions:"
echo "     source venv/bin/activate"
echo " Then run the pipeline:"
echo "     python pipeline.py --quick --skip-gnn"
echo "─────────────────────────────────────────────────────────────────────────"
