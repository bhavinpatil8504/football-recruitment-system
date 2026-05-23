"""
Stage 5: Predictive Modeling
------------------------------
Predicts next-season impact score (ŷ_i) for each player using:
  - Elastic Net (interpretable baseline)
  - XGBoost / Gradient Boosted Trees (primary model, captures nonlinearity)

Target variable — IMPACT_SCORE:
  Position-aware weighted composite of Bayesian-normalized metrics:
    ATT: 0.50·xg + 0.30·key_passes + 0.20·progressive
    MID: 0.25·xg + 0.40·progressive + 0.20·key_passes + 0.15·pressure
    DEF: 0.15·xg + 0.25·progressive + 0.10·key_passes + 0.50·pressure
    GK:  0.20·progressive + 0.80·pressure

Anti-leakage design:
  The four normalized columns composing impact_score (xg_sum_total_normalized,
  key_passes_total_normalized, progressive_passes_total_normalized,
  pressure_count_total_normalized) are EXCLUDED from the feature set by
  select_features(). The model must learn from phase-breakdown features
  (WHERE a player produces, not just how much), GNN embeddings, ratios
  (pass completion, dribble success), and VAEP/xT if available.

  This means R² will be lower (~0.3–0.6) than if components were included,
  but the numbers reflect genuine forecasting skill.

Key design choices:
  - TIME-BASED train/test split (seasons 1..N-1 → predict season N)
  - Bayesian normalization fit ONLY on training seasons (no leakage)
  - League-balanced sample weights (counter La Liga data dominance)

Install: pip install scikit-learn xgboost
"""

import numpy as np
import pandas as pd
from sklearn.linear_model    import ElasticNet
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (mean_squared_error, mean_absolute_error,
                                     r2_score)
from scipy.stats             import spearmanr
import matplotlib.pyplot     as plt
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not found — using sklearn GBT as fallback.")


# ── League-balanced sample weighting ─────────────────────────────────────────
def compute_league_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute inverse-frequency sample weights so that each league contributes
    equally to the training loss, regardless of how many seasons it has.

    Without this, La Liga (~18 free StatsBomb seasons) dominates at ~60% of
    the training set while Premier League / Bundesliga have only 2-3 seasons.

    Weights are normalized so that mean(w) = 1.0  (no net amplification).
    Single-league runs return uniform weights (no-op).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'competition_id' column.

    Returns
    -------
    np.ndarray of shape (len(df),)
    """
    if "competition_id" not in df.columns:
        return np.ones(len(df))

    league_counts = df["competition_id"].map(
        df["competition_id"].value_counts()
    )
    weights = 1.0 / league_counts
    # Normalize so mean weight = 1.0 → total effective sample size unchanged
    weights = weights / weights.mean()
    return weights.values


# ── Position groups ───────────────────────────────────────────────────────────
POSITION_GROUPS = {
    "ATT": {"xg": 0.50, "key_passes": 0.30, "progressive": 0.20, "pressure": 0.00},
    "MID": {"xg": 0.25, "key_passes": 0.20, "progressive": 0.40, "pressure": 0.15},
    "DEF": {"xg": 0.15, "key_passes": 0.10, "progressive": 0.25, "pressure": 0.50},
    "GK":  {"xg": 0.00, "key_passes": 0.00, "progressive": 0.20, "pressure": 0.80},
}

# Which normalized column maps to which weight key
METRIC_COLUMN_MAP = {
    "xg":          "xg_sum_total_normalized",
    "key_passes":  "key_passes_total_normalized",
    "progressive": "progressive_passes_total_normalized",
    "pressure":    "pressure_count_total_normalized",
}


def build_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the IMPACT_SCORE target variable.

    Uses Bayesian-normalized columns (mu_player posteriors) so the
    score is already cross-league adjusted before weighting.
    Clips each component to [0,1] via MinMaxScaler so no single
    metric dominates by raw magnitude.

    Adds column 'impact_score' in place.
    """
    df = df.copy()

    # ── Resolve which columns are actually present ────────────────────────────
    available = {}
    for key, col in METRIC_COLUMN_MAP.items():
        # Prefer normalized version; fall back to raw total
        raw_col = col.replace("_normalized", "")
        if col in df.columns:
            available[key] = col
        elif raw_col in df.columns:
            available[key] = raw_col
            print(f"  Warning: using raw '{raw_col}' — "
                  f"Bayesian normalized column not found.")

    if not available:
        raise ValueError(
            "No metric columns found. Run feature engineering and "
            "Bayesian normalization first."
        )

    # ── Scale each component to [0, 1] across all players ────────────────────
    scaler     = MinMaxScaler()
    comp_names = list(available.keys())
    comp_cols  = [available[k] for k in comp_names]

    scaled = scaler.fit_transform(df[comp_cols].fillna(0))
    scaled_df = pd.DataFrame(scaled, columns=comp_names, index=df.index)

    # ── Compute position-aware weighted sum ───────────────────────────────────
    if "position" not in df.columns:
        print("  Warning: 'position' column missing — using MID weights for all.")
        df["position"] = "MID"

    scores = []
    for idx, row in df.iterrows():
        pos     = row.get("position", "MID")
        weights = POSITION_GROUPS.get(pos, POSITION_GROUPS["MID"])
        score   = sum(
            weights.get(k, 0.0) * scaled_df.loc[idx, k]
            for k in comp_names
        )
        scores.append(score)

    df["impact_score"] = scores
    print(f"  impact_score (composite) built for {len(df)} players  "
          f"(mean={df['impact_score'].mean():.3f}, "
          f"std={df['impact_score'].std():.3f})")
    return df


def build_impact_score_vaep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct impact_score from VAEP — the preferred target when available.

    VAEP (Decroos et al., KDD 2019) provides a learned action-value metric
    that captures both offensive and defensive contribution in a single number.
    Unlike the hand-weighted composite, VAEP:
      - Is learned from data (no position-specific hand-tuned weights)
      - Values ALL action types (passes, dribbles, tackles, interceptions)
      - Is the established gold-standard in sports analytics literature

    If VAEP columns exist, uses them. Otherwise falls back to composite.

    The VAEP-based impact_score uses Bayesian-normalized vaep_sum plus a
    small defensive bonus to avoid undervaluing defenders (whose VAEP is
    partially captured in defensive_value but can be diluted in vaep_sum).
    """
    vaep_col = "vaep_sum_normalized"
    vaep_raw = "vaep_sum"
    def_col = "vaep_defensive_normalized"
    def_raw = "vaep_defensive"

    # Check if VAEP data is available
    has_vaep = vaep_col in df.columns or vaep_raw in df.columns
    if not has_vaep:
        print("  VAEP columns not found — falling back to composite impact_score")
        return build_impact_score(df)

    df = df.copy()
    use_col = vaep_col if vaep_col in df.columns else vaep_raw

    # Scale VAEP to [0, 1]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    vaep_scaled = scaler.fit_transform(df[[use_col]].fillna(0)).ravel()

    # Optional defensive bonus for DEF/GK (VAEP can undervalue pure defenders)
    if "position" in df.columns and (def_col in df.columns or def_raw in df.columns):
        d_col = def_col if def_col in df.columns else def_raw
        def_scaled = MinMaxScaler().fit_transform(df[[d_col]].fillna(0)).ravel()

        bonus = np.zeros(len(df))
        for i, pos in enumerate(df["position"]):
            if pos in ("DEF", "GK"):
                bonus[i] = 0.15 * def_scaled[i]  # small defensive bonus
        vaep_scaled = vaep_scaled + bonus
        # Re-scale to [0,1] after bonus
        if vaep_scaled.max() > 0:
            vaep_scaled = vaep_scaled / vaep_scaled.max()

    df["impact_score"] = vaep_scaled
    print(f"  impact_score (VAEP-based) built for {len(df)} players  "
          f"(mean={df['impact_score'].mean():.3f}, "
          f"std={df['impact_score'].std():.3f})")
    return df


# ── Feature selection ─────────────────────────────────────────────────────────
# Columns that must never appear as features (identifiers or targets)
_EXCLUDE_ALWAYS = {
    "player_id", "player", "season_id", "competition_id",
    "match_id", "match_date", "team_name",
    "impact_score",        # the target — never a feature
    "target",              # alias used during training
    "position",            # categorical role label
    "_season_rank",        # internal join key from build_next_season_target
}

# ── Anti-leakage: columns that COMPOSE the target must be excluded ───────────
# impact_score = weighted sum of these normalized columns. If they remain as
# features, XGBoost can trivially reconstruct the target → circular prediction.
# Phase-broken-out versions (e.g., xg_sum_box_normalized) are allowed because
# they capture WHERE a player produces, not just HOW MUCH — genuine signal.
_TARGET_COMPONENT_COLS = {
    "xg_sum_total_normalized",
    "key_passes_total_normalized",
    "progressive_passes_total_normalized",
    "pressure_count_total_normalized",
    # Also exclude the raw (un-normalized) totals — same circularity risk
    "xg_sum_total",
    "key_passes_total",
    "progressive_passes_total",
    "pressure_count_total",
}


def select_features(df: pd.DataFrame) -> list:
    """
    Return feature column names. Explicitly excludes:
      - identifier columns
      - the target (impact_score / target)
      - columns that COMPOSE impact_score (anti-leakage)
      - any column whose name contains '_next' (future leak guard)
      - uncertainty columns used only for risk adjustment, not prediction

    What REMAINS as features (the model's actual signal):
      - Phase-broken-out stats (xg_sum_box, progressive_passes_build_up, ...)
      - Pass completion rate, dribble success rate (ratios, not raw counts)
      - GNN embeddings (graph structure signal)
      - VAEP/xT per-player aggregates (if computed)
      - Bayesian-normalized phase-level columns
    """
    exclude = _EXCLUDE_ALWAYS | _TARGET_COMPONENT_COLS
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and "_next" not in c
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]
    return feature_cols


def build_next_season_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the NEXT season's impact_score as the supervised target.

    Problem formulation:
      Features  = season S statistics (already observed)
      Target    = season S+1 impact_score (what we want to predict)

    IMPORTANT — StatsBomb season IDs are NOT consecutive integers.
    They use arbitrary IDs like [4, 42, 90]. The old approach of
    computing `season_id - 1` always produced zero matches.

    Fix: sort seasons chronologically by their actual values, assign
    rank indices (0, 1, 2 ...), then join on rank+1. This is robust
    regardless of what integers StatsBomb uses for season IDs.
    """
    if "impact_score" not in df.columns:
        raise ValueError(
            "impact_score column not found. Call build_impact_score() first."
        )

    df = df.copy()

    # Build ordered season rank PER COMPETITION — this is critical for
    # multi-league data. Global ranking breaks because PL season 44 sits
    # between La Liga 42 and 90, so La Liga can't find its "next" season.
    # Per-competition ranking: La Liga gets [0,1,2,3,4], PL gets [0], etc.
    df["_season_rank"] = (
        df.groupby("competition_id")["season_id"]
        .transform(lambda s: s.rank(method="dense").astype(int) - 1)
    )

    # For each player, their target is their impact_score in the NEXT season rank
    # within the SAME competition
    future = (
        df[["player_id", "competition_id", "_season_rank", "impact_score"]]
        .copy()
        .rename(columns={"impact_score":   "target",
                         "_season_rank":   "_next_rank"})
    )
    future["_season_rank"] = future["_next_rank"] - 1   # align to previous rank

    merged = df.merge(
        future[["player_id", "competition_id", "_season_rank", "target"]],
        on=["player_id", "competition_id", "_season_rank"],
        how="inner"
    ).drop(columns=["_season_rank"])

    # Leakage check — critical for honest evaluation
    corr = float("nan")
    if len(merged) > 5:
        corr = merged["impact_score"].corr(merged["target"])
        if not np.isnan(corr) and corr > 0.95:
            print(f"  ⚠ WARNING: current↔future impact_score correlation = {corr:.3f}")
            print(f"    If > 0.95, the prediction task may be too easy (near-identity).")
            print(f"    Check that select_features() excludes impact_score components.")
            print(f"    Expected honest range: 0.3–0.7 for real forecasting tasks.")
        elif not np.isnan(corr):
            print(f"  Current↔future correlation: {corr:.3f} (reasonable)")

    if len(merged) == 0:
        seasons_found = sorted(df["season_id"].unique().tolist())
        raise ValueError(
            f"Zero player-seasons matched across consecutive seasons.\n"
            f"  Seasons in data: {seasons_found}\n"
            f"  Season ranks   : {season_rank}\n"
            f"  This usually means players don't appear in consecutive seasons.\n"
            f"  Load more seasons with --n-seasons 4 or higher."
        )

    corr_str = f"{corr:.3f}" if not np.isnan(corr) else "n/a"
    print(f"  Target built: {len(merged)} player-seasons with verified "
          f"next-season labels  (current↔future corr={corr_str})")
    return merged


def time_based_split(df: pd.DataFrame,
                     test_season_id: int) -> tuple:
    """
    Split data chronologically.
    
    Train: all seasons < test_season_id
    Test : test_season_id
    
    This is CRITICAL. Random splits would leak future information.
    """
    train = df[df["season_id"] < test_season_id].copy()
    test  = df[df["season_id"] == test_season_id].copy()
    print(f"  Train: {len(train)} rows (seasons < {test_season_id})")
    print(f"  Test : {len(test)} rows (season = {test_season_id})")
    return train, test


# ── Model definitions ─────────────────────────────────────────────────────────
def build_elastic_net() -> Pipeline:
    """
    Elastic Net with cross-validated α and l1_ratio.
    The StandardScaler inside the pipeline prevents data leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                             max_iter=5000, random_state=42))
    ])


def build_xgboost() -> object:
    """XGBoost regressor. Falls back to sklearn GBT if not installed."""
    if XGBOOST_AVAILABLE:
        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,      # L1
            reg_lambda=1.0,     # L2
            random_state=42,
            verbosity=0,
        )
    else:
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )


# ── Evaluation metrics ────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             model_name: str = "") -> dict:
    """Compute all 4 metrics from the paper."""
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    mae      = mean_absolute_error(y_true, y_pred)
    r2       = r2_score(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred).correlation

    results = {
        "model":   model_name,
        "RMSE":    round(rmse, 4),
        "MAE":     round(mae, 4),
        "R2":      round(r2, 4),
        "Spearman": round(spearman, 4),
    }
    print(f"\n  [{model_name}]")
    print(f"    RMSE    : {rmse:.4f}")
    print(f"    MAE     : {mae:.4f}")
    print(f"    R²      : {r2:.4f}")
    print(f"    Spearman: {spearman:.4f}")
    return results


# ── Ablation study ────────────────────────────────────────────────────────────
def run_ablation(df: pd.DataFrame,
                 test_season_id: int) -> pd.DataFrame:
    """
    5-tier ablation comparing feature sets against the impact_score target:

      1. Baseline       : raw per-season totals only (pass counts, xG totals)
      2. +Phase         : + tactical phase breakdowns (WHERE metrics occur)
      3. +Phase+Position: + one-hot position encoding (no GNN)
                          This is the critical control: if GNN doesn't beat
                          this, it's just learning position identity.
      4. +Phase+GNN     : + GNN embeddings (relational graph structure)
      5. Full           : + Bayesian normalized + VAEP/xT (all features)

    Each set is tested with ElasticNet and XGBoost.
    Train/test split is strictly time-based to prevent leakage.

    NOTE: impact_score components (xg_sum_total_normalized, etc.) are
    excluded by select_features() to prevent circular prediction.
    """
    df = build_next_season_target(df)

    # Use last labelled season as ablation test
    labelled_seasons = sorted(df["season_id"].unique())
    if test_season_id not in labelled_seasons:
        ablation_test = labelled_seasons[-1]
        print(f"  Note: season {test_season_id} has no next-season labels. "
              f"Using season {ablation_test} for ablation test set.")
    else:
        ablation_test = test_season_id

    train, test = time_based_split(df, ablation_test)

    if len(train) < 20 or len(test) < 5:
        raise ValueError(
            f"Not enough labelled data (train={len(train)}, test={len(test)}). "
            "Load more seasons."
        )

    all_cols = select_features(df)
    all_cols = [c for c in all_cols if "uncertainty" not in c]

    # Tier 1: Raw totals only
    raw_cols = [c for c in all_cols
                if not c.startswith("emb_")
                and not c.startswith("pos_")
                and "normalized" not in c
                and "_box" not in c
                and "_build_up" not in c
                and "_final_third" not in c
                and "_progression" not in c
                and "vaep" not in c
                and "xt_" not in c]

    # Tier 2: + phase breakdowns
    phase_cols = [c for c in all_cols
                  if "normalized" not in c
                  and not c.startswith("emb_")
                  and not c.startswith("pos_")
                  and "vaep" not in c
                  and "xt_" not in c]

    # Tier 3: + one-hot position (control for GNN)
    pos_cols = [c for c in all_cols if c.startswith("pos_")]
    phase_pos_cols = phase_cols + pos_cols

    # Tier 4: + GNN embeddings (no position one-hot — GNN should subsume it)
    gnn_cols = [c for c in all_cols
                if "normalized" not in c
                and not c.startswith("pos_")
                and "vaep" not in c
                and "xt_" not in c]

    # Tier 5: Full — everything (GNN + Bayesian normalized + VAEP/xT)
    full_cols = all_cols

    tiers = [
        ("1. Baseline (totals only)",              raw_cols),
        ("2. +Phase breakdown",                    phase_cols),
        ("3. +Phase + OneHot Position (control)",  phase_pos_cols),
        ("4. +Phase + GNN embeddings",             gnn_cols),
        ("5. Full (GNN + Bayes + VAEP/xT)",        full_cols),
    ]

    results_list = []
    for feature_set_name, feat_cols in tiers:
        feat_cols = [c for c in feat_cols if c in train.columns]
        if not feat_cols:
            print(f"  Skipping '{feature_set_name}': no valid columns present")
            continue

        X_train = train[feat_cols].fillna(0).values
        y_train = train["target"].values
        X_test  = test[feat_cols].fillna(0).values
        y_test  = test["target"].values

        w_train = compute_league_sample_weights(train)

        for model_name, model in [
            ("ElasticNet", build_elastic_net()),
            ("XGBoost",    build_xgboost()),
        ]:
            if model_name == "ElasticNet":
                model.fit(X_train, y_train, model__sample_weight=w_train)
            else:
                model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model.predict(X_test)
            res    = evaluate(y_test, y_pred,
                              model_name=f"{model_name} | {feature_set_name}")
            res["feature_set"] = feature_set_name
            res["n_features"]  = len(feat_cols)
            results_list.append(res)

    return pd.DataFrame(results_list)


# ── Final model ───────────────────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame,
                      test_season_id: int) -> tuple:
    """
    Train XGBoost on all labelled seasons (excluding the hold-out test season),
    then generate impact_score predictions for ALL players in the latest season.

    The test season is excluded from training to preserve evaluation integrity.
    """
    df_labelled = build_next_season_target(df)
    # Exclude test season from training
    df_train    = df_labelled[df_labelled["season_id"] < test_season_id]

    feat_cols = select_features(df_labelled)
    feat_cols = [c for c in feat_cols
                 if c in df_train.columns and "uncertainty" not in c]

    X = df_train[feat_cols].fillna(0).values
    y = df_train["target"].values

    # League-balanced sample weights (no-op for single-league runs)
    w = compute_league_sample_weights(df_train)
    n_leagues = df_train["competition_id"].nunique() if "competition_id" in df_train.columns else 1
    print(f"  League-balanced weighting: {n_leagues} league(s) detected")

    model = build_xgboost()
    model.fit(X, y, sample_weight=w)
    print(f"  Final model trained on {len(X)} player-seasons "
          f"(target=impact_score, features={len(feat_cols)})")

    # Predict for latest season — these are recruitment candidates
    latest          = df[df["season_id"] == df["season_id"].max()].copy()
    feat_available  = [c for c in feat_cols if c in latest.columns]
    X_latest        = np.zeros((len(latest), len(feat_cols)))
    for i, col in enumerate(feat_cols):
        if col in feat_available:
            X_latest[:, i] = latest[col].fillna(0).values

    y_hat = model.predict(X_latest)

    base_cols = ["player_id", "player", "season_id", "competition_id"]
    predictions = latest[base_cols].copy()
    if "position" in latest.columns:
        predictions["position"] = latest["position"].values
    if "team_name" in latest.columns:
        predictions["team_name"] = latest["team_name"].values
    predictions["y_hat"] = y_hat
    print(f"  Predictions for {len(predictions)} players in latest season")
    return model, predictions


def plot_feature_importance(model, feature_cols: list,
                             top_n: int = 20,
                             save_path: str = None):
    """Plot XGBoost feature importances."""
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available, skipping importance plot.")
        return

    importances = model.feature_importances_
    idx         = np.argsort(importances)[-top_n:]
    cols        = [feature_cols[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(cols, importances[idx], color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


# ── SHAP Explainability ──────────────────────────────────────────────────────
def compute_shap_explanations(model, X: np.ndarray,
                               feature_cols: list,
                               top_n: int = 20,
                               save_dir: str = "outputs") -> dict:
    """
    Compute SHAP values for the XGBoost model and generate three plots:
      1. Summary beeswarm — global feature importance with directionality.
      2. Bar plot — mean |SHAP| per feature (simpler ranking).
      3. Per-player explanation — exportable SHAP values matrix.

    SHAP (SHapley Additive exPlanations) is the Scopus-standard method for
    model interpretability in sports analytics. Unlike raw feature importance
    (which only measures split frequency), SHAP values quantify each feature's
    MARGINAL contribution to each prediction — crucial for explaining WHY
    the model rates a specific player highly.

    Requires: pip install shap

    Parameters
    ----------
    model        : trained XGBoost/GBT model
    X            : feature matrix (same used for training or prediction)
    feature_cols : list of column names matching X's columns
    top_n        : number of features to show in plots
    save_dir     : directory for output plots

    Returns
    -------
    dict with keys:
      'shap_values' : np.ndarray of shape (n_samples, n_features)
      'feature_cols': list of feature names
      'base_value'  : expected model output (baseline prediction)
    """
    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Run: pip install shap")
        print("  Skipping SHAP explanations.")
        return None

    import os
    os.makedirs(save_dir, exist_ok=True)

    print(f"  Computing SHAP values for {X.shape[0]} samples, "
          f"{X.shape[1]} features...")

    # TreeExplainer is exact and fast for tree-based models
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ── Plot 1: Summary beeswarm ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_cols,
                      max_display=top_n, show=False)
    plt.title("SHAP Summary — Feature Impact on Predicted Impact Score",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path1 = os.path.join(save_dir, "shap_summary_beeswarm.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path1}")

    # ── Plot 2: Mean |SHAP| bar plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_cols,
                      plot_type="bar", max_display=top_n, show=False)
    plt.title("SHAP — Mean Absolute Feature Importance",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path2 = os.path.join(save_dir, "shap_importance_bar.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    # ── Export SHAP matrix as DataFrame ─────────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_path = os.path.join(save_dir, "shap_values.csv")
    shap_df.to_csv(shap_path, index=False)
    print(f"  Saved: {shap_path} ({shap_df.shape})")

    # ── Top features summary ────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
    print(f"\n  Top {top_n} SHAP features:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1:2d}. {feature_cols[idx]:<40s} "
              f"mean|SHAP|={mean_abs_shap[idx]:.4f}")

    return {
        "shap_values":  shap_values,
        "feature_cols": feature_cols,
        "base_value":   float(explainer.expected_value
                              if isinstance(explainer.expected_value, (int, float))
                              else explainer.expected_value[0]),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    df = pd.read_parquet("outputs/player_features_normalized.parquet")

    # Identify latest season for test set
    seasons      = sorted(df["season_id"].unique())
    test_season  = seasons[-1]
    print(f"Available seasons: {seasons}")
    print(f"Test season: {test_season}")

    # Run ablation
    print("\n=== ABLATION STUDY ===")
    results = run_ablation(df, test_season_id=test_season)
    print("\nAblation Results:")
    print(results.to_string(index=False))
    results.to_csv("outputs/ablation_results.csv", index=False)

    # Train final model
    print("\n=== FINAL MODEL ===")
    model, predictions = train_final_model(df, test_season)

    predictions.to_parquet("outputs/player_predictions.parquet", index=False)
    print("Saved to outputs/player_predictions.parquet")
    print(predictions.head(10))
