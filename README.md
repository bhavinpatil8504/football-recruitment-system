# Machine Learning Framework for Tactical Football Recruitment

**NSUT Final Year Project** | Shikhar Singh, Abhigyan Rathi, Bhavin Mahesh Patil  
**Supervisor**: Dr. Rudresh Dwivedi | March 2026

---

## Project Structure

```
football_recruitment/
├── pipeline.py                  ← Run this to execute everything
├── requirements.txt
├── src/
│   ├── data/
│   │   └── loader.py            Stage 1: StatsBomb data loading
│   ├── features/
│   │   └── engineering.py       Stage 2: xG, tactical phase features
│   ├── models/
│   │   ├── gnn.py               Stage 3: GraphSAGE passing network
│   │   ├── bayesian.py          Stage 4: Hierarchical Bayesian normalization
│   │   └── predictor.py         Stage 5: ElasticNet + XGBoost prediction
│   └── optimization/
│       └── squad_optimizer.py   Stage 6: Risk adjustment + ILP
└── outputs/                     ← All results saved here
```

---

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### For PyTorch Geometric (GNN), install carefully:
```bash
# CPU only (laptop without GPU) — simpler
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# With CUDA GPU (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

---

## Running the Pipeline

### Quick test (10 matches, ~5 minutes — run this first)
```bash
python pipeline.py --quick --skip-gnn
```

### Full run without GNN (~15 minutes on laptop)
```bash
python pipeline.py --n-seasons 3 --skip-gnn
```

### Full run WITH GNN (~45–90 minutes on CPU, ~10 min on GPU)
```bash
python pipeline.py --n-seasons 3 --epochs 30
```

### Custom parameters
```bash
python pipeline.py \
    --n-seasons 4 \
    --lam 0.3 \
    --budget 400 \
    --formation 4-4-2 \
    --squad-size 11
```

---

## Pipeline Stages

| Stage | File | Description |
|-------|------|-------------|
| 1 | `loader.py` | Loads StatsBomb free event data (La Liga, 3 seasons) |
| 2 | `engineering.py` | Computes xG (logistic regression), phase-tagged features |
| 3 | `gnn.py` | GraphSAGE on passing networks → 32-dim player embeddings |
| 4 | `bayesian.py` | Hierarchical Bayesian cross-league normalization (PyMC) |
| 5 | `predictor.py` | Elastic Net + XGBoost, time-based split, ablation study |
| 6 | `squad_optimizer.py` | ILP squad selection (PuLP), sensitivity analysis |

---

## Key Outputs (in `outputs/`)

| File | Contents |
|------|----------|
| `ablation_results.csv` | Table comparing Baseline vs +Phase vs +GNN |
| `strategy_comparison.csv` | ILP vs Greedy vs Value-per-cost vs Random |
| `sensitivity_lambda.csv` | How squad changes with risk tolerance λ |
| `optimized_squad.parquet` | Final recommended 11-player squad |
| `player_predictions.parquet` | All players with predicted next-season impact |

---

## Data Source

This project uses **StatsBomb Open Data** (free, no API key required):
- La Liga seasons 2017/18 – 2020/21 (Messi era — richest free dataset)
- ~380 matches per season × 3 seasons = ~1,140 matches
- ~3.5 million event records

```python
from statsbombpy import sb
sb.competitions()    # list all free competitions
```

---

## Key Formulas

### Bayesian Cross-League Normalization
```
y_{p,l} ~ N(μ_p + δ_l, σ_l²)
```
- μ_p = player latent skill
- δ_l = league difficulty offset
- σ_l² = within-league variance

### Risk-Adjusted Value
```
V_adj = E[V] - λ · Var(V)
```

### ILP Objective
```
max Σ z_i · ŷ_i
s.t. Σ c_i ≤ B, positional constraints, risk cap
```

---

## GPU Notes

- **Elastic Net, XGBoost, Bayesian, ILP** — all CPU only, no GPU needed
- **GNN (Stage 3)** — runs on CPU in ~30 min for 3 seasons; GPU cuts to ~5 min
- Google Colab free T4 GPU is sufficient for the GNN stage

---

## Extending the Project (for stronger marks)

1. **Add Transfermarkt valuations** as actual transfer costs (via `soccerdata` library)
2. **Compare GCN vs GraphSAGE vs GAT** architectures in an ablation
3. **Visualize GNN embeddings** with t-SNE/UMAP — show player role clusters
4. **Add age trajectory** as a feature (players under 23 get a future value multiplier)
5. **Multi-objective optimization** — add a youth quota constraint to the ILP
