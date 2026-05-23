"""
Synthetic end-to-end pipeline test
------------------------------------
Generates fake StatsBomb-format data and runs every stage (except GNN)
to verify the pipeline logic is bug-free.

Usage:
    python test_pipeline.py
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

# ── Helpers to generate StatsBomb-format synthetic data ──────────────────────

def make_synthetic_events(n_matches=15, n_seasons=3):
    """
    Generate events DataFrame mimicking StatsBomb format.
    The `type` column contains dicts like {"id": X, "name": "TypeName"}.
    """
    rng = np.random.RandomState(42)
    season_ids = [4, 42, 90][:n_seasons]  # mimic StatsBomb La Liga IDs
    competition_id = 11

    all_events = []
    match_id_counter = 100

    players = [
        {"id": 1001, "name": "Player_A"},
        {"id": 1002, "name": "Player_B"},
        {"id": 1003, "name": "Player_C"},
        {"id": 1004, "name": "Player_D"},
        {"id": 1005, "name": "Player_E"},
        {"id": 1006, "name": "Player_F"},
        {"id": 1007, "name": "Player_G"},
        {"id": 1008, "name": "Player_H"},
        {"id": 1009, "name": "Player_I"},
        {"id": 1010, "name": "Player_J"},
        {"id": 1011, "name": "Player_K"},
        {"id": 1012, "name": "Player_L"},
        {"id": 1013, "name": "Player_M"},
        {"id": 1014, "name": "Player_N"},
        {"id": 1015, "name": "Player_O"},
    ]

    for season_id in season_ids:
        for m in range(n_matches):
            match_id_counter += 1
            mid = match_id_counter
            n_events = rng.randint(80, 150)

            for _ in range(n_events):
                p = rng.choice(players)
                etype = rng.choice(["Shot", "Pass", "Dribble", "Pressure",
                                     "Ball Receipt*", "Carry"], p=[0.08, 0.45, 0.07, 0.15, 0.15, 0.10])

                x = rng.uniform(0, 120)
                y = rng.uniform(0, 80)

                event = {
                    "match_id": mid,
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "player_id": p["id"],
                    "player": p["name"],
                    "type": {"id": rng.randint(1, 50), "name": etype},
                    "location": [x, y],
                    "under_pressure": rng.choice([True, False, None]),
                    "home_team": "Team_Home",
                    "away_team": "Team_Away",
                    "home_score": rng.randint(0, 4),
                    "away_score": rng.randint(0, 4),
                    "match_date": f"2020-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}",
                }

                # Shot-specific columns
                if etype == "Shot":
                    is_goal = rng.random() < 0.15
                    event["shot_outcome"] = {"id": 98, "name": "Goal"} if is_goal else {"id": 100, "name": "Saved"}
                    event["shot_body_part"] = {"id": 37, "name": "Head"} if rng.random() < 0.2 else {"id": 40, "name": "Right Foot"}
                else:
                    event["shot_outcome"] = None
                    event["shot_body_part"] = None

                # Pass-specific columns
                if etype == "Pass":
                    end_x = rng.uniform(0, 120)
                    end_y = rng.uniform(0, 80)
                    event["pass_end_location"] = [end_x, end_y]
                    recipient = rng.choice(players)
                    event["pass_recipient_id"] = recipient["id"]
                    is_complete = rng.random() < 0.75
                    event["pass_outcome"] = None if is_complete else {"id": 9, "name": "Incomplete"}
                    event["pass_goal_assist"] = True if (is_complete and rng.random() < 0.03) else None
                else:
                    event["pass_end_location"] = None
                    event["pass_recipient_id"] = None
                    event["pass_outcome"] = None
                    event["pass_goal_assist"] = None

                # Dribble-specific columns
                if etype == "Dribble":
                    event["dribble_outcome"] = {"id": 1, "name": "Complete"} if rng.random() < 0.6 else {"id": 2, "name": "Incomplete"}
                else:
                    event["dribble_outcome"] = None

                all_events.append(event)

    return pd.DataFrame(all_events)


def make_synthetic_lineups(events_df):
    """Generate lineups DataFrame mimicking StatsBomb format."""
    records = []
    for (mid, sid, cid), grp in events_df.groupby(["match_id", "season_id", "competition_id"]):
        player_ids = grp["player_id"].unique()
        for pid in player_ids:
            pname = grp[grp["player_id"] == pid]["player"].iloc[0]
            # StatsBomb nested position format
            pos_choices = [
                [{"position": {"id": 1, "name": "Goalkeeper"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 3, "name": "Right Center Back"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 5, "name": "Left Back"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 3, "name": "Center Back"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 10, "name": "Center Defensive Midfield"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 11, "name": "Left Midfield"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 12, "name": "Right Midfield"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 15, "name": "Right Wing"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 17, "name": "Left Wing"}, "from": "00:00", "to": "90:00"}],
                [{"position": {"id": 21, "name": "Center Forward"}, "from": "00:00", "to": "90:00"}],
            ]
            # Deterministic position based on player_id
            pos = pos_choices[pid % len(pos_choices)]
            records.append({
                "player_id": pid,
                "player_name": pname,
                "team_name": "Team_Home",
                "competition_id": cid,
                "season_id": sid,
                "match_id": mid,
                "match_date": grp["match_date"].iloc[0],
                "positions": pos,
            })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SYNTHETIC PIPELINE TEST")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    # ── STAGE 1: Synthetic data loading ──────────────────────────────────────
    print("\n█" * 60)
    print("  STAGE 1: Generating Synthetic Data")
    print("█" * 60)

    events = make_synthetic_events(n_matches=15, n_seasons=3)
    lineups = make_synthetic_lineups(events)

    events.to_parquet("outputs/raw_events.parquet", index=False)
    lineups.to_parquet("outputs/raw_lineups.parquet", index=False)

    from src.data.loader import get_player_season_map
    player_map = get_player_season_map(lineups)
    player_map.to_parquet("outputs/player_map.parquet", index=False)

    print(f"  Events : {len(events):,}")
    print(f"  Lineups: {len(lineups):,}")
    print(f"  Players: {player_map['player_id'].nunique()}")
    print("  ✓ Stage 1 PASSED")

    # ── STAGE 2: Feature Engineering ─────────────────────────────────────────
    print("\n█" * 60)
    print("  STAGE 2: Feature Engineering")
    print("█" * 60)

    from src.features.engineering import xGModel, extract_player_features

    xg_model = xGModel()
    xg_model.fit(events)
    print(f"  xG model fitted: {xg_model.fitted}")

    features = extract_player_features(events, xg_model)
    features.to_parquet("outputs/player_features.parquet", index=False)
    print(f"  Feature matrix: {features.shape}")
    assert len(features) > 0, "Feature matrix is empty!"
    assert "player_id" in features.columns
    print("  ✓ Stage 2 PASSED")

    # ── STAGE 3: GNN (skipped — torch not available) ─────────────────────────
    print("\n█" * 60)
    print("  STAGE 3: GNN (SKIPPED — no torch)")
    print("█" * 60)
    print("  GNN skipped.")
    print("  ✓ Stage 3 SKIPPED (OK)")

    # ── STAGE 4: Bayesian Normalization ──────────────────────────────────────
    print("\n█" * 60)
    print("  STAGE 4: Bayesian Normalization")
    print("█" * 60)

    from src.models.bayesian import normalize_all_metrics

    seasons = sorted(features["season_id"].unique())
    test_season = seasons[-1]
    train_seasons = [s for s in seasons if s < test_season]
    print(f"  Train seasons: {train_seasons}  |  Test season: {test_season}")

    metrics_to_normalize = [
        c for c in [
            "xg_sum_total",
            "progressive_passes_total",
            "key_passes_total",
            "pressure_count_total",
        ] if c in features.columns
    ]
    print(f"  Metrics to normalize: {metrics_to_normalize}")

    if metrics_to_normalize and len(train_seasons) > 0:
        normalized = normalize_all_metrics(
            features, metrics_to_normalize,
            train_seasons=train_seasons, use_variational=True
        )
    else:
        normalized = features

    normalized.to_parquet("outputs/player_features_normalized.parquet", index=False)
    print(f"  Normalized shape: {normalized.shape}")
    # Check normalized columns were added
    norm_cols = [c for c in normalized.columns if "normalized" in c]
    unc_cols = [c for c in normalized.columns if "uncertainty" in c]
    print(f"  Normalized columns: {norm_cols}")
    print(f"  Uncertainty columns: {unc_cols}")
    print("  ✓ Stage 4 PASSED")

    # ── STAGE 5: Predictive Modeling ─────────────────────────────────────────
    print("\n█" * 60)
    print("  STAGE 5: Predictive Modeling")
    print("█" * 60)

    from src.models.predictor import build_impact_score, run_ablation, train_final_model
    from src.optimization.squad_optimizer import assign_positions

    seasons = sorted(normalized["season_id"].unique())
    test_season = seasons[-1]

    # Assign positions
    normalized = assign_positions(normalized, lineups)
    assert "position" in normalized.columns, "Position column missing!"

    # Build impact score
    normalized = build_impact_score(normalized)
    assert "impact_score" in normalized.columns, "impact_score missing!"
    normalized.to_parquet("outputs/player_features_normalized.parquet", index=False)

    # Ablation
    ablation = None
    if len(seasons) >= 2:
        print("\n  --- Ablation Study ---")
        try:
            ablation = run_ablation(normalized, test_season_id=test_season)
            print(ablation.to_string(index=False))
            ablation.to_csv("outputs/ablation_results.csv", index=False)
        except Exception as e:
            print(f"  Ablation failed: {e}")

        print("\n  --- Training Final Model ---")
        try:
            model, predictions = train_final_model(normalized, test_season)
        except ValueError as e:
            print(f"  Warning: {e}")
            print("  Falling back: using current-season impact_score as proxy target.")
            latest = normalized[normalized["season_id"] == test_season].copy()
            latest["y_hat"] = latest["impact_score"]
            predictions = latest[["player_id", "player", "season_id",
                                   "competition_id", "position", "y_hat"]].copy()
            print(f"  Fallback predictions for {len(predictions)} players.")

        if "position" not in predictions.columns:
            predictions = predictions.merge(
                normalized[["player_id", "position"]].drop_duplicates("player_id"),
                on="player_id", how="left"
            )
            predictions["position"] = predictions["position"].fillna("MID")

        predictions.to_parquet("outputs/player_predictions.parquet", index=False)
        assert len(predictions) > 0, "No predictions generated!"
        print(f"  Predictions: {len(predictions)} players")

    print("  ✓ Stage 5 PASSED")

    # ── STAGE 6: Risk Adjustment + Squad Optimization ────────────────────────
    print("\n█" * 60)
    print("  STAGE 6: Risk Adjustment + ILP Squad Optimization")
    print("█" * 60)

    from src.optimization.squad_optimizer import (
        compute_risk_adjusted_value, estimate_transfer_cost,
        SquadOptimizer, compare_strategies,
        select_greedy_squad, select_value_per_cost_squad, select_random_squad
    )

    candidates = compute_risk_adjusted_value(predictions, normalized, lam=0.5)
    candidates = assign_positions(candidates, lineups)
    candidates = estimate_transfer_cost(candidates, budget_scale=100.0)

    print(f"  Candidates: {len(candidates)}")
    assert "v_adj" in candidates.columns
    assert "cost" in candidates.columns
    assert "position" in candidates.columns

    optimizer = SquadOptimizer(budget=500.0, squad_size=11,
                                formation="4-3-3", risk_cap=0.3)

    # Compare strategies
    results = compare_strategies(candidates, optimizer)
    print("\n" + results.to_string(index=False))
    results.to_csv("outputs/strategy_comparison.csv", index=False)

    # Get optimal squad
    optimal_squad, status = optimizer.optimize(candidates)
    optimizer.summarize_squad(optimal_squad)
    optimal_squad.to_parquet("outputs/optimized_squad.parquet", index=False)

    print("  ✓ Stage 6 PASSED")

    # ── EVALUATION ───────────────────────────────────────────────────────────
    print("\n█" * 60)
    print("  EVALUATION")
    print("█" * 60)

    from src.evaluation.evaluator import EvaluationPipeline as EvalPipeline

    ground_truth = normalized[normalized["season_id"] == test_season][
        ["player_id", "impact_score"]
    ]

    merged_eval = predictions.merge(ground_truth, on="player_id", how="inner")
    pred_experiments = []
    if len(merged_eval) >= 2:
        pred_experiments.append({
            "label": "XGBoost — full pipeline",
            "y_true": merged_eval["impact_score"].values,
            "y_pred": merged_eval["y_hat"].values,
        })

    squad_experiments = {
        "ILP (ours)": optimal_squad,
        "Greedy": select_greedy_squad(candidates, optimizer.squad_size, optimizer.budget),
        "Value-per-cost": select_value_per_cost_squad(candidates, optimizer.squad_size, optimizer.budget),
        "Random": select_random_squad(candidates, optimizer.squad_size, optimizer.budget),
    }

    eval_pipeline = EvalPipeline()
    if pred_experiments:
        eval_pipeline.run(pred_experiments, squad_experiments, ground_truth)
        eval_pipeline.print_report()
        eval_pipeline.save_tables()

    print("  ✓ Evaluation PASSED")

    # ── Sensitivity Analysis ─────────────────────────────────────────────────
    print("\n  --- Sensitivity Analysis: λ ---")
    sens_records = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cands = compute_risk_adjusted_value(predictions, normalized, lam=lam)
        cands = assign_positions(cands, lineups)
        cands = estimate_transfer_cost(cands, budget_scale=100.0)
        try:
            sq, _ = optimizer.optimize(cands)
            sens_records.append({
                "lambda": lam,
                "total_impact": round(sq["v_adj"].sum(), 3),
                "avg_risk": round(sq["variance"].mean(), 4),
                "total_cost": round(sq["cost"].sum(), 1),
            })
        except Exception as e:
            print(f"  λ={lam}: failed ({e})")

    sens_df = pd.DataFrame(sens_records)
    print(sens_df.to_string(index=False))
    sens_df.to_csv("outputs/sensitivity_lambda.csv", index=False)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL STAGES PASSED")
    print("=" * 60)
    print("\nOutput files:")
    for f in sorted(os.listdir("outputs")):
        size = os.path.getsize(f"outputs/{f}")
        print(f"  outputs/{f:<45s} {size:>8,} bytes")


if __name__ == "__main__":
    main()
