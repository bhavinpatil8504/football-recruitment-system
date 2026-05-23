# Football Recruitment ML Pipeline — Project Instructions

## What this is
7-stage ML football recruitment pipeline (B.Tech Project / BTP) simulating Moneyball-style squad building for a financially constrained club using StatsBomb open data. Built by Shikhar (NSUT CSE).

## Pipeline stages
1. **Data Loading** — Multi-league StatsBomb events from top-5 European leagues + Champions League + international tournaments
2. **Feature Engineering** — VAEP (Decroos et al., KDD 2019) + xT (Singh 2019) via socceraction, xG (logistic regression), per-player-season stats with tactical phase breakdowns (build-up, progression, final third, box)
3. **GNN** — GAT/GraphSAGE on passing networks with self-supervised link prediction (4-head GAT, 64 hidden, 32 embedding dim) + t-SNE visualization + one-hot position baseline for ablation control
4. **Bayesian Cross-League Normalization** — Empirical Bayes / James-Stein shrinkage with per-league difficulty offsets + quantile compression for dominant-club bias
5. **XGBoost Predictive Model** — Predicts next-season impact_score (VAEP-based when available) with time-based split, 5-tier ablation study, league-balanced sample weighting, SHAP explainability. Anti-leakage: impact_score components excluded from features.
5b. **Transfer Validation** — Retrospective experiment: identifies real transfers in data, compares pre-transfer model predictions to realized post-transfer performance. Held-out league generalization test (train on N-1 leagues, predict held-out).
6. **ILP Squad Optimization** — PuLP CBC solver with budget, formation, risk penalty, max-per-club, youth sustainability constraints + elite filtering + Transfermarkt-calibrated cost tiers
7. **Tactical Blueprint** — 8-dimensional data-derived style profiles. Learned player×system interaction model when transfer data is sufficient (≥20 transfers), otherwise additive heuristic fallback. Cross-style comparison across 5 preset styles.

## Key files
- `pipeline.py` — End-to-end orchestrator (run this)
- `src/data/loader.py` — Stage 1: StatsBomb data ingestion
- `src/features/engineering.py` — Stage 2: VAEP + xT + xG + feature extraction
- `src/models/gnn.py` — Stage 3: GAT/GraphSAGE + t-SNE visualization + position baseline
- `src/models/bayesian.py` — Stage 4: Empirical Bayes normalizer
- `src/models/predictor.py` — Stage 5: XGBoost + 5-tier ablation + SHAP + anti-leakage
- `src/optimization/squad_optimizer.py` — Stage 6: Risk-adjusted ILP
- `src/optimization/squad_optimizer_tactical.py` — Stage 7: Tactical ILP
- `src/tactics/blueprint.py` — Tactical profiles + learned interaction model
- `src/evaluation/evaluator.py` — Evaluation pipeline
- `src/evaluation/transfer_validation.py` — Stage 5b: retrospective transfer validation + held-out league

## How to run

### Setup (once)
```
# Windows
setup.bat

# Mac/Linux
chmod +x setup.sh && ./setup.sh

# Fallback
python -m venv venv
venv\Scripts\activate   # or: source venv/bin/activate
python setup.py
```

### Run pipeline
```
# Activate venv first
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux

# Quick test (~5 min, low RAM)
python pipeline.py --quick --skip-gnn --n-seasons 3

# Standard run (~45 min, ~6 GB RAM)
python pipeline.py --n-seasons 10 --skip-gnn

# Full run with GNN (~1-2 hours, ~8 GB RAM)
python pipeline.py --n-seasons 10

# FULL multi-league (RECOMMENDED for final results, needs 12-16 GB RAM)
python pipeline.py --n-seasons 10 --multi-league --n-cl-seasons 5 --min-young-pct 0.3 --budget 80 --style gegenpressing
```

## Key design decisions
- **VAEP as primary target**: When socceraction is installed, impact_score is built from VAEP (learned action valuation) instead of hand-weighted composite. This prevents the circularity where XGBoost predicts a function of its own input features.
- **Anti-leakage**: impact_score component columns (xg_sum_total_normalized, etc.) are explicitly excluded from the feature set by select_features(). The silent fallback (y_hat = impact_score) has been replaced with a hard-fail.
- **5-tier ablation**: Baseline → +Phase → +Phase+OneHotPosition (control) → +Phase+GNN → Full (GNN+Bayes+VAEP/xT). The position control tests whether GNN adds signal beyond position identity.
- **Transfer validation**: Identifies real club-to-club transfers in the data and compares pre-transfer model predictions to realized post-transfer performance. This is the "money figure" validating the framework's central claim.
- **Held-out league**: Train on N-1 leagues, predict held-out league. Tests cross-league generalization.
- **Learned tactical model**: When ≥20 transfers available, trains a GBR on player_features × system_features to predict post-transfer performance. Replaces additive heuristic. Falls back to V_style = V_adj + α·(fit-0.5)·spread otherwise.
- **Risk adjustment**: V_adj = ŷ − λ·risk_index with 3-source composite index (NOT Markowitz mean-variance). Honest framing as "robust scoring with risk penalty."
- **Elite filtering**: Removes top 15% performers and players from ~20 richest clubs
- **League balancing**: Inverse-frequency sample weights in XGBoost
- **Quantile compression**: Winsorize top 5% per league before Bayesian normalization

## Known issues being debugged
- Some competitions cause "unable to allocate" memory errors on machines with <12 GB RAM
- La Liga has ~18 StatsBomb seasons vs 2-3 for other leagues (addressed via sample weighting)
- The pipeline needs internet access (StatsBomb data is downloaded live each run)

## What we need from this run
Run Option D or E from RUN_INSTRUCTIONS.txt on a machine with 16 GB+ RAM. After completion, copy back the **entire `outputs/` folder** — it contains all results, plots, and data files needed for the BTP report and research paper.

## Output files to expect
- `outputs/ablation_results.csv` — 5-tier feature ablation (Table 1)
- `outputs/strategy_comparison.csv` — ILP vs greedy vs random (Table 2)
- `outputs/transfer_validation.png` — Retrospective transfer validation (Figure 5)
- `outputs/transfer_validation_results.csv` — Transfer validation detail
- `outputs/held_out_league_results.csv` — Cross-league generalization (Table 3)
- `outputs/gnn_embeddings_tsne.png` — GNN role cluster visualization (Figure 4)
- `outputs/shap_summary_beeswarm.png` — SHAP feature importance (Figure 3)
- `outputs/shap_importance_bar.png` — SHAP bar plot
- `outputs/shap_values.csv` — Raw SHAP values
- `outputs/interaction_model_metrics.json` — Learned interaction model stats
- `outputs/optimized_squad.parquet` — Impact-only squad
- `outputs/tactical_squad.parquet` — Style-adjusted squad
- `outputs/tactical_style_comparison.csv` — Cross-style analysis (Table 4)
- `outputs/league_effects_*.png` — Bayesian league difficulty plots
- `outputs/sensitivity_*.csv` — Lambda and alpha sensitivity analysis
- `outputs/vaep_xt_features.parquet` — VAEP/xT per-player aggregates
- `outputs/eval_report.png` — Evaluation summary
- Various `.parquet` intermediate files

## If something fails
1. If memory error: reduce `--n-seasons` to 5, or add `--skip-gnn`, or drop `--multi-league`
2. If a specific competition fails to load: the pipeline continues with remaining competitions
3. If GNN fails: add `--skip-gnn` flag (pipeline still works, just without graph embeddings)
4. If XGBoost not installed: falls back to sklearn GradientBoostingRegressor automatically
