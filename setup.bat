@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  setup.bat  —  Windows setup for football_recruitment
REM  Run once from inside the project folder:  .\setup.bat
REM
REM  Prerequisites: Python 3.10 or 3.11 installed and on PATH
REM  Check with:    python --version
REM ─────────────────────────────────────────────────────────────────────────

echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10 or 3.11 from python.org
    pause & exit /b 1
)

echo.
echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 ( echo ERROR creating venv & pause & exit /b 1 )

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Step 4: Installing numpy + pandas + scipy + pyarrow FIRST (foundation layer)...
pip install "numpy==1.26.4" "pandas==2.1.4" "scipy==1.11.4" "pyarrow>=14.0.0"
if errorlevel 1 ( echo ERROR at Step 4 & pause & exit /b 1 )

echo.
echo Step 5: Installing scikit-learn and xgboost...
pip install "scikit-learn==1.3.2" "xgboost==2.0.3"
if errorlevel 1 ( echo ERROR at Step 5 & pause & exit /b 1 )

echo.
echo Step 6: Installing StatsBomb data tools...
pip install "statsbombpy>=1.18.0" "socceraction==1.4.2"
if errorlevel 1 ( echo ERROR at Step 6 & pause & exit /b 1 )

echo.
echo Step 7: Installing SHAP (model explainability)...
pip install shap
if errorlevel 1 ( echo WARNING: SHAP install failed - non-critical & echo. )

echo.
echo Step 8: Installing PuLP (optimization)...
pip install "pulp==2.7.0"
if errorlevel 1 ( echo ERROR at Step 8 & pause & exit /b 1 )

echo.
echo Step 9: Installing visualization and utilities...
pip install "matplotlib==3.8.2" "seaborn==0.13.2" "plotly==5.18.0" "tqdm==4.66.1" "joblib==1.3.2"
if errorlevel 1 ( echo ERROR at Step 9 & pause & exit /b 1 )

echo.
echo Step 10: Installing PyTorch (CPU version)...
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 ( echo ERROR at Step 10 & pause & exit /b 1 )

echo.
echo Step 11: Installing PyTorch Geometric (GNN)...
pip install torch-geometric==2.4.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv ^
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
if errorlevel 1 (
    echo WARNING: PyG extras failed. Trying minimal install...
    pip install torch-geometric==2.4.0
)

echo.
echo Step 12: Verifying installation...
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"
python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import sklearn; print('sklearn', sklearn.__version__)"
python -c "import xgboost; print('xgboost', xgboost.__version__)"
python -c "import statsbombpy; print('statsbombpy OK')"
python -c "import socceraction; print('socceraction OK')"
python -c "import shap; print('shap', shap.__version__)"
python -c "import pulp; print('pulp', pulp.__version__)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"

echo.
echo ─────────────────────────────────────────────────────────────────────────
echo  Setup complete.
echo  To activate this environment in future sessions:
echo      venv\Scripts\activate
echo  Then run the pipeline:
echo      python pipeline.py --quick --skip-gnn
echo ─────────────────────────────────────────────────────────────────────────
pause
