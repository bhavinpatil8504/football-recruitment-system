"""
End-to-End Recruitment Pipeline
---------------------------------
Run this single file to execute all 7 stages in order.

Usage:
    python pipeline.py                      # full run (may take ~1 hour)
    python pipeline.py --quick              # fast test (10 matches, no GNN)
    python pipeline.py --skip-gnn           # skip GNN, use raw features only
    python pipeline.py --lam 0.3 --budget 400
    python pipeline.py --style gegenpressing # tactical style selection

Stages:
    1. Data loading    (StatsBomb open data)
    2. Feature engineering (xG, xT, VAEP, tactical phases)
    3. GNN embeddings  (passing network representation learning)
    4. Bayesian normalization (cross-league adjustment)
    5. Predictive modeling (ElasticNet + XGBoost)
    6. Risk adjustment + ILP squad optimization
    7. Tactical blueprint — style-aware squad optimization (NOVELTY)
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np

# Make src importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader         import quick_load_laliga, get_player_season_map
from src.features.engineering import (xGModel, extract_player_features,
                                     compute_vaep, compute_xt,
                                     aggregate_vaep_xt,
                                     SOCCERACTION_AVAILABLE)
from src.models.bayesian     import normalize_all_metrics
from src.models.predictor    import (build_impact_score, build_impact_score_vaep,
                                     run_ablation, train_final_model)
from src.optimization.squad_optimizer import (
    compute_risk_adjusted_value, assign_positions,
    estimate_transfer_cost, filter_elite_players,
    SquadOptimizer, compare_strategies,
    plot_strategy_comparison
)
from src.evaluation.evaluator import EvaluationPipeline

os.makedirs("outputs", exist_ok=True)


def print_banner(stage: str) -> None:
    print("\n" + "█"*60)
    print(f"  {stage}")
    print("█"*60)


def parse_args():
    parser = argparse.ArgumentParser(description="Football Recruitment Pipeline")
    parser.add_argument("--quick",     action="store_true",
                        help="Quick test mode (10 matches per season)")
    parser.add_argument("--skip-gnn",  action="store_true",
                        help="Skip GNN training (faster, lower accuracy)")
    parser.add_argument("--n-seasons", type=int, default=10,
                        help="Number of La Liga seasons to load (default: 10)")
    parser.add_argument("--multi-league", action="store_true",
                        help="Include all top-5 European leagues + Champions League")
    parser.add_argument("--n-cl-seasons", type=int, default=5,
                        help="Number of Champions League seasons to load (default: 5, max: 14)")
    parser.add_argument("--lam",       type=float, default=0.5,
                        help="Risk aversion parameter λ (default: 0.5)")
    parser.add_argument("--budget",    type=float, default=80.0,
                        help="Transfer budget in €M (default: 80 — mid-table La Liga club)")
    parser.add_argument("--formation", type=str, default="4-3-3",
                        help="Formation (4-3-3, 4-4-2, 3-5-2)")
    parser.add_argument("--squad-size", type=int, default=11,
                        help="Squad size to select (default: 11)")
    parser.add_argument("--epochs",    type=int, default=30,
                        help="GNN training epochs (default: 30)")
    parser.add_argument("--gnn-arch",  type=str, default="gat",
                        choices=["gat", "sage"],
                        help="GNN architecture: gat or sage (default: gat)")
    parser.add_argument("--style",     type=str, default="gegenpressing",
                        help="Tactical style preset (gegenpressing, positional_play, "
                             "low_block_counter, pragmatic, balanced)")
    parser.add_argument("--max-per-club", type=int, default=3,
                        help="Max players from any single club (default: 3). "
                             "Enforces recruitment diversity.")
    parser.add_argument("--alpha",     type=float, default=0.4,
                        help="Tactical fit weight α (0=pure impact, 1=pure style)")
    parser.add_argument("--elite-pct", type=float, default=85.0,
                        help="Remove top (100 - elite_pct)%% players as unreachable "
                             "(default: 85 = remove top 15%%)")
    parser.add_argument("--no-elite-filter", action="store_true",
                        help="Disable elite club and percentile filtering")
    parser.add_argument("--min-young-pct", type=float, default=0.0,
                        help="Min fraction of squad that must be young prospects "
                             "(0.0-1.0, default: 0.0 = no constraint). "
                             "E.g., 0.3 = at least 30%% must be young.")
    return parser.parse_args()


def main():
    args = parse_args()
    t0   = time.time()

    max_matches = 10 if args.quick else None

    # Torch is only needed for GNN — lazy import
    if not args.skip_gnn:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            print("  Warning: torch not installed — forcing --skip-gnn")
            args.skip_gnn = True
            device = "cpu"
    else:
        device = "cpu"

    print(f"\nDevice: {device}")
    print(f"Quick mode   : {args.quick}")
    print(f"Skip GNN     : {args.skip_gnn}")
    print(f"GNN arch     : {args.gnn_arch}")
    print(f"Multi-league : {args.multi_league}")
    if args.multi_league:
        print(f"CL seasons   : {args.n_cl_seasons}")
    print(f"Elite filter : {'OFF' if args.no_elite_filter else f'top {100-args.elite_pct:.0f}% removed'}")
    print(f"Budget       : €{args.budget:.0f}M")
    print(f"Tactical style: {args.style}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: DATA LOADING
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("STAGE 1: Data Loading")

    from src.data.loader import (load_multi_season,
                                   TOP5_LEAGUE_PAIRS, CHAMPIONS_LEAGUE_PAIRS,
                                   INTERNATIONAL_PAIRS)

    data = quick_load_laliga(
        n_seasons=args.n_seasons,
        max_matches_per_season=max_matches   # None = all matches
    )

    # ── Multi-league: add top-5 European leagues + CL + internationals ────
    if args.multi_league:
        print("\n  Loading top-5 league data (PL, Bundesliga, Serie A, Ligue 1)...")
        # Top-5 leagues (excl. La Liga which is already loaded)
        extra_pairs = list(TOP5_LEAGUE_PAIRS)
        # Champions League — rich multi-national player pool
        extra_pairs += CHAMPIONS_LEAGUE_PAIRS[:args.n_cl_seasons]
        # International tournaments
        extra_pairs += list(INTERNATIONAL_PAIRS)

        loaded_count = 0
        for comp_id, season_id in extra_pairs:
            try:
                print(f"\n  Loading competition={comp_id}, season={season_id}...")
                extra = load_multi_season(
                    [(comp_id, season_id)],
                    max_matches_per_season=max_matches
                )
                data["events"]  = pd.concat([data["events"], extra["events"]],
                                             ignore_index=True)
                data["lineups"] = pd.concat([data["lineups"], extra["lineups"]],
                                             ignore_index=True)
                loaded_count += 1
            except Exception as e:
                print(f"  Warning: skipping comp={comp_id}/season={season_id}: {e}")

        print(f"\n  Multi-league: loaded {loaded_count}/{len(extra_pairs)} "
              f"additional competition-seasons")
        print(f"  Total events: {len(data['events']):,}")

    # data is a dict {"events": DataFrame, "lineups": DataFrame}
    events  = data["events"]
    lineups = data["lineups"]

    if not isinstance(events, pd.DataFrame) or len(events) == 0:
        print("ERROR: No events loaded. Check your internet connection.")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    events.to_parquet("outputs/raw_events.parquet",   index=False)
    lineups.to_parquet("outputs/raw_lineups.parquet", index=False)

    player_map = get_player_season_map(lineups)
    player_map.to_parquet("outputs/player_map.parquet", index=False)

    print(f"\n  Events : {len(events):,}")
    print(f"  Lineups: {len(lineups):,}")
    print(f"  Players: {player_map['player_id'].nunique():,}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("STAGE 2: Feature Engineering (xG, VAEP, xT, Phases)")

    xg_model = xGModel()
    xg_model.fit(events)

    features = extract_player_features(events, xg_model)

    # ── VAEP + xT (action-level valuation via socceraction) ──────────────────
    if SOCCERACTION_AVAILABLE:
        print("\n  --- VAEP (Decroos et al., KDD 2019) ---")
        try:
            vaep_df = compute_vaep(events)
            xt_df = compute_xt(events)
            vaep_xt_agg = aggregate_vaep_xt(vaep_df, xt_df)

            if len(vaep_xt_agg) > 0:
                # Merge VAEP/xT features into player features
                merge_cols = ["player_id", "season_id", "competition_id"]
                features = features.merge(vaep_xt_agg, on=merge_cols, how="left")
                features[vaep_xt_agg.columns.difference(merge_cols)] = (
                    features[vaep_xt_agg.columns.difference(merge_cols)].fillna(0)
                )
                print(f"  VAEP/xT merged — feature matrix now: {features.shape}")
                vaep_xt_agg.to_parquet("outputs/vaep_xt_features.parquet", index=False)
            else:
                print("  VAEP/xT: no data produced — continuing without")
        except Exception as e:
            print(f"  VAEP/xT failed: {e}")
            print("  Continuing with xG-based features only")
    else:
        print("\n  socceraction not installed — skipping VAEP/xT")
        print("  Install with: pip install socceraction==1.4.2")

    features.to_parquet("outputs/player_features.parquet", index=False)
    print(f"\n  Feature matrix: {features.shape}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: REPRESENTATION LEARNING (GNN)
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("STAGE 3: GNN Representation Learning")

    skip_cols    = ["player_id", "player", "season_id", "competition_id"]
    feature_cols = [c for c in features.columns
                    if c not in skip_cols
                    and features[c].dtype in [np.float64, np.float32,
                                              np.int64, np.int32]]

    if not args.skip_gnn and len(feature_cols) > 0:
        import torch
        from src.models.gnn import (build_all_match_graphs, train_gnn,
                                     extract_player_embeddings)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = features.copy()
        features_scaled[feature_cols] = scaler.fit_transform(
            features[feature_cols].fillna(0)
        )

        graphs = build_all_match_graphs(events, features_scaled, feature_cols)
        if graphs:
            gnn_model = train_gnn(
                graphs,
                in_channels=len(feature_cols),
                hidden_dim=64, embed_dim=32,
                epochs=args.epochs,
                device=device,
                architecture=args.gnn_arch
            )
            embeddings = extract_player_embeddings(gnn_model, graphs, device)
            embeddings.to_parquet("outputs/player_embeddings.parquet", index=False)
            torch.save(gnn_model.state_dict(), "outputs/gnn_model.pt")

            # Merge embeddings into features
            features = features.merge(embeddings, on="player_id", how="left")
            features.to_parquet("outputs/player_features_with_emb.parquet",
                                index=False)

            # Embedding visualization (t-SNE)
            try:
                from src.models.gnn import visualize_embeddings
                visualize_embeddings(embeddings, features, lineups,
                                    save_dir="outputs", method="tsne")
            except Exception as e:
                print(f"  Embedding visualization skipped: {e}")
        else:
            print("  No valid graphs — skipping GNN.")
    else:
        print("  GNN skipped.")

    # Always add one-hot position baseline columns (needed for Tier 3 ablation)
    try:
        from src.models.gnn import build_position_baseline
        pos_baseline = build_position_baseline(features, lineups)
        pos_cols_existing = [c for c in features.columns if c.startswith("pos_")]
        if not pos_cols_existing:
            features = features.merge(pos_baseline, on="player_id", how="left")
            for c in ["pos_GK", "pos_DEF", "pos_MID", "pos_ATT"]:
                if c in features.columns:
                    features[c] = features[c].fillna(0)
            print(f"  Added one-hot position baseline (for ablation control)")
    except Exception as e:
        print(f"  Position baseline skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: BAYESIAN NORMALIZATION
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("STAGE 4: Bayesian Cross-League Normalization")

    seasons      = sorted(features["season_id"].unique())
    test_season  = seasons[-1]
    train_seasons = [s for s in seasons if s < test_season]

    print(f"  Train seasons: {train_seasons}  |  Test season: {test_season}")

    metrics_to_normalize = [
        c for c in [
            "xg_sum_total",
            "progressive_passes_total",
            "key_passes_total",
            "pressure_count_total",
            # VAEP/xT — normalize if available (from socceraction)
            "vaep_sum",
            "vaep_offensive",
            "vaep_defensive",
            "xt_sum",
        ] if c in features.columns
    ]

    if metrics_to_normalize and len(train_seasons) > 0:
        normalized = normalize_all_metrics(
            features,
            metrics_to_normalize,
            train_seasons=train_seasons,   # leakage fix: fit on train only
        )
    else:
        print("  No metrics / seasons for normalization — using raw features.")
        normalized = features

    normalized.to_parquet("outputs/player_features_normalized.parquet",
                          index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5: PREDICTIVE MODELING
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("STAGE 5: Predictive Modeling")

    seasons      = sorted(normalized["season_id"].unique())
    test_season  = seasons[-1]
    print(f"  Seasons available: {seasons}")
    print(f"  Test season (held out): {test_season}")

    # Assign positions FIRST — needed by build_impact_score and Stage 6
    normalized = assign_positions(normalized, lineups)

    # Build impact_score — prefer VAEP-based (learned) over composite (hand-weighted)
    has_vaep = "vaep_sum" in normalized.columns or "vaep_sum_normalized" in normalized.columns
    if has_vaep:
        print("  Using VAEP-based impact_score (learned action valuation)")
        normalized = build_impact_score_vaep(normalized)
    else:
        print("  VAEP not available — using composite impact_score")
        normalized = build_impact_score(normalized)
    normalized.to_parquet("outputs/player_features_normalized.parquet",
                          index=False)

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
            # Cross-season join failed — this is a FATAL problem for prediction validity.
            # The old fallback silently set y_hat = impact_score (current season),
            # making all downstream results (ablation, SHAP, squads) meaningless.
            # Now we hard-fail so the user knows to load more seasons.
            print(f"\n  FATAL: {e}")
            print("  ┌─────────────────────────────────────────────────────────────┐")
            print("  │  Cannot build prediction targets. This usually means       │")
            print("  │  too few seasons loaded for cross-season player matching.  │")
            print("  │  Try: --n-seasons 5 or higher                              │")
            print("  │  The old silent fallback (y_hat = impact_score) has been   │")
            print("  │  removed because it invalidated all downstream results.    │")
            print("  └─────────────────────────────────────────────────────────────┘")
            raise SystemExit(1)

        # Carry position column into predictions
        if "position" not in predictions.columns:
            predictions = predictions.merge(
                normalized[["player_id", "position"]].drop_duplicates("player_id"),
                on="player_id", how="left"
            )
            predictions["position"] = predictions["position"].fillna("MID")

        # Carry team_name from lineups into predictions (needed for club diversity constraint)
        if "team_name" not in predictions.columns and "team_name" in lineups.columns:
            team_map = (
                lineups.groupby("player_id")["team_name"]
                .agg(lambda x: x.value_counts().index[0])   # most common team
                .reset_index()
            )
            predictions = predictions.merge(team_map, on="player_id", how="left")
            predictions["team_name"] = predictions["team_name"].fillna("Unknown")
            print(f"  Teams in candidate pool: {predictions['team_name'].nunique()}")

        predictions.to_parquet("outputs/player_predictions.parquet", index=False)

        # ── SHAP Explainability ──────────────────────────────────────────────
        # Generates SHAP beeswarm + bar plots for model interpretability.
        # Required for Scopus-grade publication (see research document).
        if 'model' in dir():
            try:
                from src.models.predictor import compute_shap_explanations, select_features
                shap_feat_cols = select_features(normalized)
                shap_feat_cols = [c for c in shap_feat_cols
                                  if c in normalized.columns and "uncertainty" not in c]
                latest_for_shap = normalized[
                    normalized["season_id"] == test_season
                ].copy()
                X_shap = latest_for_shap[shap_feat_cols].fillna(0).values
                shap_result = compute_shap_explanations(
                    model, X_shap, shap_feat_cols,
                    top_n=20, save_dir="outputs"
                )
                if shap_result:
                    print("  SHAP explanations generated successfully.")
            except Exception as e:
                print(f"  SHAP analysis skipped: {e}")
    else:
        print("  Need ≥2 seasons for prediction. Load more data.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5b: TRANSFER VALIDATION + HELD-OUT LEAGUE EXPERIMENT
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("STAGE 5b: Retrospective Transfer Validation")

    from src.evaluation.transfer_validation import (
        identify_transfers, validate_transfers,
        run_held_out_league_experiment
    )

    # Detect real transfers in the data
    player_map = get_player_season_map(lineups)
    transfers = identify_transfers(player_map, normalized, events=events)
    transfer_metrics = {}

    if len(transfers) > 0:
        transfer_metrics = validate_transfers(
            transfers, predictions, normalized,
            save_dir="outputs"
        )
        if transfer_metrics:
            print(f"\n  Transfer validation summary:")
            for k, v in transfer_metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
    else:
        print("  No transfers detected (need multi-season data with club changes)")

    # Held-out league experiment (if multi-league)
    if normalized["competition_id"].nunique() > 1:
        print("\n  --- Held-Out League Generalization ---")
        held_out_results = run_held_out_league_experiment(
            features, normalized,
            build_impact_fn=build_impact_score_vaep if has_vaep else build_impact_score,
            save_dir="outputs"
        )
    else:
        print("  Single-league data — skipping held-out league experiment")
        print("  Run with --multi-league for cross-league generalization tests")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 6: RISK ADJUSTMENT + SQUAD OPTIMIZATION
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("STAGE 6: Risk Adjustment + ILP Squad Optimization")

    candidates = compute_risk_adjusted_value(
        predictions, normalized, lam=args.lam
    )

    # Position is already in predictions/candidates from Stage 5.
    # assign_positions is called only as a safety net for any missing values.
    candidates = assign_positions(candidates, lineups)

    # ── Elite player filtering ────────────────────────────────────────────────
    # A financially constrained club cannot sign top-tier stars or poach from
    # elite clubs. Remove unreachable players before cost estimation so that
    # the cost tiers calibrate to the ACTUAL available pool.
    if not args.no_elite_filter:
        candidates = filter_elite_players(
            candidates,
            elite_percentile=args.elite_pct,
            exclude_elite_clubs=True,
        )

    candidates = estimate_transfer_cost(candidates)

    print(f"\n  Candidates pool : {len(candidates)} players")
    print(f"  Budget          : {args.budget}")
    print(f"  Formation       : {args.formation}")
    print(f"  λ (risk)        : {args.lam}")

    optimizer = SquadOptimizer(
        budget=args.budget,
        squad_size=args.squad_size,
        formation=args.formation,
        risk_cap=0.3,
        max_per_club=args.max_per_club,
        min_young_pct=args.min_young_pct,
    )

    # ── Compare all strategies ─────────────────────────────────────────────────
    results = compare_strategies(candidates, optimizer)
    print("\n" + results.to_string(index=False))
    results.to_csv("outputs/strategy_comparison.csv", index=False)

    # Get and display the optimal squad
    optimal_squad, _ = optimizer.optimize(candidates)
    optimizer.summarize_squad(optimal_squad)
    optimal_squad.to_parquet("outputs/optimized_squad.parquet", index=False)

    # ── Unified evaluation: link prediction accuracy → squad outcomes ──────────
    print_banner("EVALUATION: Linking Predictive Performance to Squad Outcomes")

    ground_truth = normalized[normalized["season_id"] == test_season][
        ["player_id", "impact_score"]
    ]

    # Prediction experiments (one per feature set from ablation if available)
    pred_experiments = []
    if ablation is not None and "y_true" in ablation.columns:
        pass   # ablation already computed; rebuild experiments if needed
    # Minimal: evaluate final model predictions vs. ground truth
    merged_eval = predictions.merge(ground_truth, on="player_id", how="inner")
    if len(merged_eval) >= 2:
        pred_experiments.append({
            "label":  "XGBoost — full pipeline",
            "y_true": merged_eval["impact_score"].values,
            "y_pred": merged_eval["y_hat"].values,
        })

    # Squad experiments (all strategies)
    from src.optimization.squad_optimizer import (
        select_greedy_squad, select_value_per_cost_squad, select_random_squad
    )
    mpc = args.max_per_club
    squad_experiments = {
        "ILP (ours)":      optimal_squad,
        "Greedy":          select_greedy_squad(candidates, optimizer.squad_size,
                                               optimizer.budget, max_per_club=mpc),
        "Value-per-cost":  select_value_per_cost_squad(candidates, optimizer.squad_size,
                                                        optimizer.budget, max_per_club=mpc),
        "Random":          select_random_squad(candidates, optimizer.squad_size,
                                                optimizer.budget, max_per_club=mpc),
    }

    eval_pipeline = EvaluationPipeline()
    if pred_experiments:
        eval_pipeline.run(pred_experiments, squad_experiments, ground_truth)
        eval_pipeline.print_report()
        eval_pipeline.plot_report(save_path="outputs/eval_report.png")
        eval_pipeline.save_tables()

    # ── Sensitivity analysis on λ ─────────────────────────────────────────────
    print("\n  --- Sensitivity Analysis: λ ---")
    sens_records = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cands = compute_risk_adjusted_value(predictions, normalized, lam=lam)
        cands = assign_positions(cands, lineups)
        if not args.no_elite_filter:
            cands = filter_elite_players(
                cands, elite_percentile=args.elite_pct,
                exclude_elite_clubs=True,
            )
        cands = estimate_transfer_cost(cands)
        try:
            sq, _ = optimizer.optimize(cands)
            sens_records.append({
                "lambda": lam,
                "total_impact": round(sq["v_adj"].sum(), 3),
                "avg_risk":     round(sq["variance"].mean(), 4),
                "total_cost":   round(sq["cost"].sum(), 1),
            })
        except Exception as e:
            print(f"  λ={lam}: failed ({e})")

    sens_df = pd.DataFrame(sens_records)
    print(sens_df.to_string(index=False))
    sens_df.to_csv("outputs/sensitivity_lambda.csv", index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 7: TACTICAL BLUEPRINT — STYLE-AWARE RECRUITMENT
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("STAGE 7: Tactical Blueprint — Style-Aware Recruitment")

    from src.tactics.blueprint import (
        extract_tactical_profiles, compute_tactical_fit,
        compute_position_aware_fit, compute_style_adjusted_value,
        train_style_interaction_model, compute_style_adjusted_value_learned,
        PRESET_STYLES, TACTICAL_DIMS
    )
    from src.optimization.squad_optimizer_tactical import TacticalSquadOptimizer

    # Step 1: Extract tactical profiles from event data
    print("\n  --- Extracting Tactical Profiles ---")
    tactical_profiles = extract_tactical_profiles(events, normalized)
    tactical_profiles.to_parquet("outputs/tactical_profiles.parquet", index=False)

    # Step 2: Show profile summary
    print(f"\n  Tactical profile means across all players:")
    for dim in TACTICAL_DIMS:
        print(f"    {dim:<28s}: {tactical_profiles[dim].mean():.3f}")

    # Step 3: Train learned style interaction model (if transfer data available)
    interaction_model = None
    if len(transfers) >= 20:
        print("\n  --- Training Style Interaction Model ---")
        try:
            interaction_model = train_style_interaction_model(
                transfers, tactical_profiles, normalized,
                target_col="impact_score"
            )
            if interaction_model:
                import json
                metrics_path = "outputs/interaction_model_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(interaction_model["metrics"], f, indent=2)
                print(f"  Saved: {metrics_path}")
        except Exception as e:
            print(f"  Interaction model failed: {e}")
            print("  Falling back to additive heuristic")
    else:
        print(f"\n  {len(transfers)} transfers detected (need 20+ for learned model)")
        print("  Using additive heuristic for style adjustment")

    # Step 4: Compute tactical fit for the chosen style
    style_name = args.style
    if style_name not in PRESET_STYLES:
        print(f"  Warning: unknown style '{style_name}', using 'gegenpressing'")
        style_name = "gegenpressing"

    target_style = PRESET_STYLES[style_name]
    print(f"\n  --- Target Style: {style_name} ---")
    for dim in TACTICAL_DIMS:
        print(f"    {dim:<28s}: {target_style[dim]:.2f}")

    # Position-aware fit (weights tactical dims differently per position)
    if "position" in candidates.columns:
        pos_map = candidates.set_index("player_id")["position"].to_dict()
        aligned_positions = tactical_profiles["player_id"].map(pos_map).fillna("MID")
        tactical_fit = compute_position_aware_fit(
            tactical_profiles, target_style, aligned_positions
        )
    else:
        tactical_fit = compute_tactical_fit(tactical_profiles, target_style)

    # Step 5: Build style-adjusted squads
    print(f"\n  --- Style-Adjusted Squad Optimization (α={args.alpha}) ---")
    if interaction_model is not None:
        print("  Using LEARNED interaction model (player × system features)")
        styled_candidates = compute_style_adjusted_value_learned(
            candidates, interaction_model, tactical_profiles,
            target_style, alpha=args.alpha
        )
    else:
        print("  Using additive heuristic (no learned interaction)")
        styled_candidates = compute_style_adjusted_value(
            candidates, tactical_fit, alpha=args.alpha
        )

    tac_optimizer = TacticalSquadOptimizer(
        budget=args.budget,
        squad_size=args.squad_size,
        formation=args.formation,
        risk_cap=0.3,
        max_per_club=args.max_per_club
    )

    tac_squad, tac_status = tac_optimizer.optimize(styled_candidates, value_col="v_style")
    print(f"  Solver status: {tac_status}")
    tac_optimizer.summarize_tactical_squad(tac_squad, style_name=style_name)
    tac_squad.to_parquet("outputs/tactical_squad.parquet", index=False)

    # Step 5: Compare across all preset styles
    print("\n  --- Cross-Style Comparison ---")
    style_records = []
    for sname, svec in PRESET_STYLES.items():
        fit = compute_tactical_fit(tactical_profiles, svec)
        styled = compute_style_adjusted_value(candidates, fit, alpha=args.alpha)
        try:
            sq, st = tac_optimizer.optimize(styled, value_col="v_style")
            style_records.append({
                "style":        sname,
                "status":       st,
                "total_v_adj":  round(sq["v_adj"].sum(), 4),
                "mean_fit":     round(sq["tactical_fit"].mean(), 4),
                "total_v_style":round(sq["v_style"].sum(), 4),
                "total_cost":   round(sq["cost"].sum(), 1),
            })
        except Exception as e:
            print(f"  {sname}: failed ({e})")

    style_df = pd.DataFrame(style_records)
    print(style_df.to_string(index=False))
    style_df.to_csv("outputs/tactical_style_comparison.csv", index=False)

    # Step 6: Alpha sensitivity — how much does tactical emphasis change the squad?
    print("\n  --- Sensitivity Analysis: α (tactical weight) ---")
    alpha_records = []
    fit_for_chosen = compute_tactical_fit(tactical_profiles, target_style)
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        styled = compute_style_adjusted_value(candidates, fit_for_chosen, alpha=alpha)
        try:
            sq, _ = tac_optimizer.optimize(styled, value_col="v_style")
            alpha_records.append({
                "alpha":       alpha,
                "total_v_adj": round(sq["v_adj"].sum(), 4),
                "mean_fit":    round(sq["tactical_fit"].mean(), 4),
                "total_cost":  round(sq["cost"].sum(), 1),
            })
        except Exception as e:
            print(f"  α={alpha}: failed ({e})")

    alpha_df = pd.DataFrame(alpha_records)
    print(alpha_df.to_string(index=False))
    alpha_df.to_csv("outputs/sensitivity_alpha.csv", index=False)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "="*60)
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  All outputs saved to: outputs/")
    print("="*60)
    print("\nKey output files:")
    print("  outputs/ablation_results.csv          ← Table 1: 5-tier feature ablation")
    print("  outputs/strategy_comparison.csv       ← Table 2: ILP vs greedy vs random")
    print("  outputs/transfer_validation.png       ← Figure 5: retrospective transfers")
    print("  outputs/transfer_validation_results.csv ← Transfer validation detail")
    print("  outputs/held_out_league_results.csv   ← Table 3: cross-league generalization")
    print("  outputs/gnn_embeddings_tsne.png       ← Figure 4: GNN role clusters")
    print("  outputs/shap_summary_beeswarm.png     ← Figure 3: SHAP feature importance")
    print("  outputs/sensitivity_lambda.csv        ← Sensitivity analysis (risk)")
    print("  outputs/sensitivity_alpha.csv         ← Sensitivity analysis (tactical)")
    print("  outputs/tactical_style_comparison.csv ← Table 4: cross-style comparison")
    print("  outputs/interaction_model_metrics.json← Learned interaction model stats")
    print("  outputs/tactical_squad.parquet        ← Final squad (style-adjusted)")
    print("  outputs/optimized_squad.parquet       ← Squad (impact-only baseline)")


if __name__ == "__main__":
    main()
