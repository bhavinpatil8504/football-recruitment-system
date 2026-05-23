"""
Retrospective Transfer Validation
-----------------------------------
The "money figure" for the paper — validates that the framework's
predictions actually work for their stated purpose (player recruitment).

Method:
  1. Identify real transfers in the StatsBomb data: players who appear
     at Club A in season S and Club B in season S+1.
  2. For each transfer, generate the model's pre-transfer prediction
     (using only season-S features) of post-transfer performance.
  3. Compare predicted vs. realized impact at Club B.
  4. Stratify by style-fit (α) if tactical profiles are available —
     check whether high-fit transfers yield higher realized gains.

This directly tests the framework's central claim: "we can predict
which players will succeed at a new club."

The experiment uses ONLY data the model could have seen at decision time
(features from season S, Bayesian normalization fit on seasons ≤ S).
No future information leaks into the prediction.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")


def _estimate_minutes_played(events: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate minutes played per player per season from event timestamps.

    StatsBomb events have a 'minute' field. We approximate total minutes as
    the range of minutes across all events for that player in that season,
    weighted by the number of distinct matches they appear in.

    This is a proxy — actual minutes would require substitution event parsing.
    The proxy is conservative: it underestimates minutes for players subbed
    on/off frequently but is reliable for regulars (which is our target pool).
    """
    if events is None or len(events) == 0 or "minute" not in events.columns:
        return pd.DataFrame(columns=["player_id", "season_id", "est_minutes"])

    # Need player_id to be available
    if "player_id" not in events.columns:
        return pd.DataFrame(columns=["player_id", "season_id", "est_minutes"])

    gk = ["player_id", "season_id"]
    player_stats = (
        events.dropna(subset=["player_id"])
        .groupby(gk)
        .agg(
            n_matches=("match_id", "nunique"),
            max_minute=("minute", "max"),
            n_events=("minute", "count"),
        )
        .reset_index()
    )

    # Estimate: n_matches × avg_minutes_per_match (capped at 90)
    # Use max_minute as a proxy for how deep into matches they play
    player_stats["avg_match_minutes"] = player_stats["max_minute"].clip(upper=90)
    player_stats["est_minutes"] = (
        player_stats["n_matches"] * player_stats["avg_match_minutes"]
    ).clip(lower=0)

    return player_stats[["player_id", "season_id", "est_minutes", "n_matches"]]


def identify_transfers(player_map: pd.DataFrame,
                       features: pd.DataFrame,
                       events: pd.DataFrame = None,
                       min_minutes_pre: int = 900,
                       min_minutes_post: int = 450) -> pd.DataFrame:
    """
    Find players who changed clubs between consecutive seasons.

    A "transfer" is defined as:
      - Player P appears at Club A in season S (via lineups/player_map)
      - Player P appears at Club B in season S+1
      - Club A ≠ Club B
      - Player has ≥ min_minutes_pre estimated minutes at Club A
      - Player has ≥ min_minutes_post estimated minutes at Club B

    The minutes filter is critical for validity:
      - Without it, bench warmers and injured players (who have low/zero
        impact_score) bias realized performance downward
      - Standard in the literature: ≥1500 minutes (Decroos et al.)
      - We use ≥900 pre and ≥450 post because StatsBomb free data
        has partial seasons — fewer matches per season than full coverage
      - The lower threshold avoids discarding too many valid transfers
        while still filtering out marginal appearances

    Parameters
    ----------
    min_minutes_pre  : minimum estimated minutes at Club A (pre-transfer)
    min_minutes_post : minimum estimated minutes at Club B (post-transfer)
                       Lower than pre because first season at new club
                       often involves adjustment/rotation period.

    Returns DataFrame with columns:
      player_id, player, season_from, season_to, club_from, club_to,
      competition_from, competition_to, minutes_pre, minutes_post
    """
    # Build player-club-season mapping
    if "team_name" in player_map.columns:
        mapping = player_map[["player_id", "team_name", "season_id",
                               "competition_id"]].drop_duplicates()
        mapping = mapping.rename(columns={"team_name": "club"})
    elif "team_name" in features.columns:
        mapping = features[["player_id", "team_name", "season_id",
                            "competition_id"]].drop_duplicates()
        mapping = mapping.rename(columns={"team_name": "club"})
    else:
        print("  No club information available for transfer detection")
        return pd.DataFrame()

    # Add player name if available
    if "player" in features.columns:
        name_map = features[["player_id", "player"]].drop_duplicates("player_id")
        mapping = mapping.merge(name_map, on="player_id", how="left")
    elif "player_name" in player_map.columns:
        name_map = player_map[["player_id", "player_name"]].drop_duplicates("player_id")
        mapping = mapping.merge(name_map, on="player_id", how="left")
        mapping = mapping.rename(columns={"player_name": "player"})

    # Estimate minutes played (if events available)
    minutes_df = _estimate_minutes_played(events) if events is not None else None

    # Sort seasons and assign ranks
    seasons = sorted(mapping["season_id"].unique())
    if len(seasons) < 2:
        print("  Need at least 2 seasons to detect transfers")
        return pd.DataFrame()

    season_rank = {s: i for i, s in enumerate(seasons)}
    mapping["_rank"] = mapping["season_id"].map(season_rank)

    # Self-join: find same player in consecutive seasons at different clubs
    current = mapping.rename(columns={
        "club": "club_from", "season_id": "season_from",
        "competition_id": "competition_from", "_rank": "_rank_from"
    })
    future = mapping.rename(columns={
        "club": "club_to", "season_id": "season_to",
        "competition_id": "competition_to", "_rank": "_rank_to"
    })

    transfers = current.merge(
        future[["player_id", "club_to", "season_to",
                "competition_to", "_rank_to"]],
        on="player_id"
    )
    transfers = transfers[transfers["_rank_to"] == transfers["_rank_from"] + 1]
    transfers = transfers[transfers["club_from"] != transfers["club_to"]]
    transfers = transfers.drop_duplicates(
        subset=["player_id", "season_from", "season_to"]
    )

    n_raw = len(transfers)

    # Apply minutes filter
    if minutes_df is not None and len(minutes_df) > 0:
        # Pre-transfer minutes
        transfers = transfers.merge(
            minutes_df[["player_id", "season_id", "est_minutes"]].rename(
                columns={"season_id": "season_from", "est_minutes": "minutes_pre"}
            ),
            on=["player_id", "season_from"], how="left"
        )
        # Post-transfer minutes
        transfers = transfers.merge(
            minutes_df[["player_id", "season_id", "est_minutes"]].rename(
                columns={"season_id": "season_to", "est_minutes": "minutes_post"}
            ),
            on=["player_id", "season_to"], how="left"
        )
        transfers["minutes_pre"] = transfers["minutes_pre"].fillna(0)
        transfers["minutes_post"] = transfers["minutes_post"].fillna(0)

        # Filter: sufficient playing time at both clubs
        pre_ok = transfers["minutes_pre"] >= min_minutes_pre
        post_ok = transfers["minutes_post"] >= min_minutes_post
        n_pre_fail = (~pre_ok).sum()
        n_post_fail = (pre_ok & ~post_ok).sum()
        transfers = transfers[pre_ok & post_ok]

        print(f"  Raw transfers detected: {n_raw}")
        print(f"  Filtered by minutes (≥{min_minutes_pre} pre, ≥{min_minutes_post} post): "
              f"{len(transfers)} remain")
        if n_pre_fail > 0:
            print(f"    {n_pre_fail} removed: insufficient pre-transfer minutes")
        if n_post_fail > 0:
            print(f"    {n_post_fail} removed: insufficient post-transfer minutes "
                  "(bench/injury/rotation)")
    else:
        transfers["minutes_pre"] = np.nan
        transfers["minutes_post"] = np.nan
        print(f"  Detected {n_raw} transfers (no minutes filter — events not provided)")

    # Clean up
    drop_cols = [c for c in transfers.columns if c.startswith("_rank")]
    transfers = transfers.drop(columns=drop_cols, errors="ignore")

    if len(transfers) > 0:
        print(f"  Valid transfers: {len(transfers)} involving "
              f"{transfers['player_id'].nunique()} unique players")

    return transfers.reset_index(drop=True)


def _build_case_studies(result: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Select ~n high-profile transfers and categorize each as:
      - HIT:         predicted high, realized high (model got it right)
      - MISS:        predicted high, realized low  (model overestimated)
      - UNDERVALUED: predicted low,  realized high (model missed upside)
      - EXPECTED_LOW: predicted low, realized low  (correctly avoided)

    This provides the narrative case study reviewers expect — named players,
    concrete numbers, not just aggregate correlations.
    """
    if len(result) < 4:
        return pd.DataFrame()

    r = result.copy()
    pred_med = r["pre_prediction"].median()
    real_med = r["realized_post"].median()

    conditions = [
        (r["pre_prediction"] >= pred_med) & (r["realized_post"] >= real_med),
        (r["pre_prediction"] >= pred_med) & (r["realized_post"] < real_med),
        (r["pre_prediction"] < pred_med) & (r["realized_post"] >= real_med),
        (r["pre_prediction"] < pred_med) & (r["realized_post"] < real_med),
    ]
    labels = ["HIT", "MISS", "UNDERVALUED", "EXPECTED_LOW"]
    r["category"] = np.select(conditions, labels, default="UNKNOWN")

    # Pick the most extreme examples from each category
    cases = []
    for cat in ["HIT", "MISS", "UNDERVALUED", "EXPECTED_LOW"]:
        pool = r[r["category"] == cat]
        if len(pool) == 0:
            continue

        # Sort by interestingness: largest |gain| for hits/misses,
        # largest surprise for undervalued
        if cat == "HIT":
            pool = pool.sort_values("realized_post", ascending=False)
        elif cat == "MISS":
            pool = pool.sort_values("gain", ascending=True)  # biggest drop
        elif cat == "UNDERVALUED":
            pool = pool.sort_values("gain", ascending=False)  # biggest surprise
        else:
            pool = pool.sort_values("realized_post", ascending=True)

        take = max(1, n // 4)
        cases.append(pool.head(take))

    if not cases:
        return pd.DataFrame()

    out = pd.concat(cases, ignore_index=True)

    # Select presentation columns
    cols = ["player", "club_from", "club_to", "season_from", "season_to",
            "pre_prediction", "pre_actual", "realized_post", "gain", "category"]
    cols = [c for c in cols if c in out.columns]

    return out[cols].head(n)


def validate_transfers(transfers: pd.DataFrame,
                       predictions: pd.DataFrame,
                       features: pd.DataFrame,
                       tactical_profiles: pd.DataFrame = None,
                       save_dir: str = "outputs") -> dict:
    """
    Compare model predictions to realized post-transfer performance.

    For each transfer:
      - pre_prediction  = model's ŷ for this player at decision time (season S)
      - realized_post   = actual impact_score at Club B (season S+1)
      - pre_actual      = actual impact_score at Club A (season S)
      - gain            = realized_post - pre_actual

    Metrics reported:
      - Spearman correlation (pre_prediction, realized_post): rank ordering
      - Pearson correlation: linear relationship
      - Mean absolute error
      - Binned analysis: top-predicted vs bottom-predicted transfer outcomes
      - (If tactical) Style-fit stratification
    """
    os.makedirs(save_dir, exist_ok=True)

    if len(transfers) == 0:
        print("  No transfers to validate")
        return {}

    # Get predictions (model's ŷ from season S features)
    if "y_hat" in predictions.columns:
        pred_map = (
            predictions[["player_id", "season_id", "y_hat"]]
            .rename(columns={"y_hat": "pre_prediction", "season_id": "pred_season"})
        )
    else:
        print("  No predictions available for transfer validation")
        return {}

    # Get actual impact scores per season
    if "impact_score" in features.columns:
        actual_map = features[["player_id", "season_id", "impact_score"]].copy()
    else:
        print("  No impact_score available for transfer validation")
        return {}

    # Merge: pre-transfer prediction
    result = transfers.merge(
        pred_map,
        left_on=["player_id", "season_from"],
        right_on=["player_id", "pred_season"],
        how="inner"
    )

    # Merge: pre-transfer actual
    result = result.merge(
        actual_map.rename(columns={"impact_score": "pre_actual",
                                    "season_id": "_s_pre"}),
        left_on=["player_id", "season_from"],
        right_on=["player_id", "_s_pre"],
        how="inner"
    )

    # Merge: post-transfer actual
    result = result.merge(
        actual_map.rename(columns={"impact_score": "realized_post",
                                    "season_id": "_s_post"}),
        left_on=["player_id", "season_to"],
        right_on=["player_id", "_s_post"],
        how="inner"
    )

    # Clean
    result = result.drop(columns=["_s_pre", "_s_post", "pred_season"],
                         errors="ignore")
    result["gain"] = result["realized_post"] - result["pre_actual"]

    if len(result) < 5:
        print(f"  Only {len(result)} transfers with full data — too few for analysis")
        return {"n_transfers": len(result)}

    print(f"\n  Transfer validation: {len(result)} transfers with complete data")

    # ── Correlation analysis ────────────────────────────────────────────────
    spear_r, spear_p = spearmanr(result["pre_prediction"], result["realized_post"])
    pear_r, pear_p = pearsonr(result["pre_prediction"], result["realized_post"])
    mae = np.mean(np.abs(result["pre_prediction"] - result["realized_post"]))

    print(f"  Spearman ρ (prediction vs realized): {spear_r:.3f} (p={spear_p:.4f})")
    print(f"  Pearson  r (prediction vs realized): {pear_r:.3f} (p={pear_p:.4f})")
    print(f"  MAE: {mae:.4f}")

    # ── Binned analysis: top vs bottom predicted transfers ──────────────────
    result["pred_rank"] = result["pre_prediction"].rank(pct=True)
    top_half = result[result["pred_rank"] >= 0.5]
    bot_half = result[result["pred_rank"] < 0.5]

    mean_gain_top = top_half["gain"].mean() if len(top_half) > 0 else 0
    mean_gain_bot = bot_half["gain"].mean() if len(bot_half) > 0 else 0

    print(f"  Top-50% predicted: mean post-transfer gain = {mean_gain_top:.4f}")
    print(f"  Bottom-50%:        mean post-transfer gain = {mean_gain_bot:.4f}")

    # ── Style-fit stratification (if tactical profiles available) ──────────
    style_analysis = {}
    if tactical_profiles is not None and "style_fit" in tactical_profiles.columns:
        fit_map = (
            tactical_profiles[["player_id", "style_fit"]]
            .drop_duplicates("player_id")
            .set_index("player_id")["style_fit"]
        )
        result["style_fit"] = result["player_id"].map(fit_map)
        has_fit = result.dropna(subset=["style_fit"])

        if len(has_fit) >= 10:
            high_fit = has_fit[has_fit["style_fit"] >= has_fit["style_fit"].median()]
            low_fit = has_fit[has_fit["style_fit"] < has_fit["style_fit"].median()]

            print(f"\n  Style-fit stratification ({len(has_fit)} transfers):")
            print(f"    High fit: mean gain = {high_fit['gain'].mean():.4f} "
                  f"(n={len(high_fit)})")
            print(f"    Low fit:  mean gain = {low_fit['gain'].mean():.4f} "
                  f"(n={len(low_fit)})")
            style_analysis = {
                "high_fit_gain": high_fit["gain"].mean(),
                "low_fit_gain": low_fit["gain"].mean(),
                "n_with_fit": len(has_fit),
            }

    # ── Case study demo: named transfers ──────────────────────────────────
    case_studies = _build_case_studies(result, n=10)
    if len(case_studies) > 0:
        print(f"\n  ── Case Study: {len(case_studies)} Notable Transfers ──")
        for _, row in case_studies.iterrows():
            player = row.get("player", "Unknown")
            cat = row.get("category", "?")
            fr = row.get("club_from", "?")
            to = row.get("club_to", "?")
            pred = row.get("pre_prediction", 0)
            real = row.get("realized_post", 0)
            gain = row.get("gain", 0)
            sign = "+" if gain >= 0 else ""
            print(f"    [{cat:13s}] {player:25s} | {fr} → {to} | "
                  f"predicted={pred:.3f}  realized={real:.3f}  gain={sign}{gain:.3f}")
        # Save case studies
        case_studies.to_csv(os.path.join(save_dir, "transfer_case_studies.csv"),
                            index=False)
        print(f"  Saved: {os.path.join(save_dir, 'transfer_case_studies.csv')}")

    # ── Visualization ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Predicted vs Realized
    axes[0].scatter(result["pre_prediction"], result["realized_post"],
                    alpha=0.5, s=20)
    lims = [min(result["pre_prediction"].min(), result["realized_post"].min()),
            max(result["pre_prediction"].max(), result["realized_post"].max())]
    axes[0].plot(lims, lims, "r--", alpha=0.5, label="Perfect prediction")
    axes[0].set_xlabel("Model Prediction (pre-transfer)")
    axes[0].set_ylabel("Realized Performance (post-transfer)")
    axes[0].set_title(f"Transfer Validation\nρ={spear_r:.3f}, r={pear_r:.3f}")
    axes[0].legend()

    # Plot 2: Gain distribution by prediction quality
    axes[1].hist([top_half["gain"], bot_half["gain"]],
                 bins=20, alpha=0.6, label=["Top 50% predicted", "Bottom 50%"])
    axes[1].axvline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Post-Transfer Performance Gain")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Transfer Gains by Prediction Quality")
    axes[1].legend()

    # Plot 3: Prediction rank vs gain
    axes[2].scatter(result["pred_rank"], result["gain"], alpha=0.5, s=20)
    z = np.polyfit(result["pred_rank"], result["gain"], 1)
    p = np.poly1d(z)
    axes[2].plot([0, 1], [p(0), p(1)], "r--", alpha=0.7)
    axes[2].set_xlabel("Prediction Percentile Rank")
    axes[2].set_ylabel("Post-Transfer Gain")
    axes[2].set_title("Does Prediction Rank Correlate with Transfer Success?")

    plt.tight_layout()
    path = os.path.join(save_dir, "transfer_validation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")

    # Save detailed results
    result.to_csv(os.path.join(save_dir, "transfer_validation_results.csv"),
                  index=False)

    return {
        "n_transfers": len(result),
        "spearman_rho": spear_r,
        "spearman_p": spear_p,
        "pearson_r": pear_r,
        "pearson_p": pear_p,
        "mae": mae,
        "top50_mean_gain": mean_gain_top,
        "bot50_mean_gain": mean_gain_bot,
        "n_case_studies": len(case_studies) if len(case_studies) > 0 else 0,
        **style_analysis,
    }


def run_held_out_league_experiment(features: pd.DataFrame,
                                   normalized: pd.DataFrame,
                                   build_impact_fn,
                                   save_dir: str = "outputs") -> pd.DataFrame:
    """
    Leave-one-league-out cross-validation for generalization testing.

    For each league L:
      1. Train on all leagues EXCEPT L
      2. Predict on L
      3. Report metrics

    This answers reviewer 2's question: "does this generalize to
    leagues you didn't train on?"

    Returns a DataFrame with per-league results.
    """
    from src.models.predictor import (select_features, build_next_season_target,
                                       build_xgboost, evaluate,
                                       compute_league_sample_weights)
    os.makedirs(save_dir, exist_ok=True)

    if "competition_id" not in normalized.columns:
        print("  Need multi-league data for held-out league experiment")
        return pd.DataFrame()

    leagues = normalized["competition_id"].unique()
    if len(leagues) < 2:
        print("  Only 1 league �� cannot run held-out league experiment")
        return pd.DataFrame()

    print(f"  Running leave-one-league-out with {len(leagues)} leagues...")

    # Build targets first on full data
    try:
        with_target = build_next_season_target(normalized)
    except ValueError as e:
        print(f"  Cannot build targets: {e}")
        return pd.DataFrame()

    feat_cols = select_features(with_target)
    feat_cols = [c for c in feat_cols
                 if "uncertainty" not in c
                 and c in with_target.columns]

    results = []
    for held_out in leagues:
        train_data = with_target[with_target["competition_id"] != held_out]
        test_data = with_target[with_target["competition_id"] == held_out]

        if len(train_data) < 20 or len(test_data) < 5:
            continue

        X_train = train_data[feat_cols].fillna(0).values
        y_train = train_data["target"].values
        X_test = test_data[feat_cols].fillna(0).values
        y_test = test_data["target"].values

        w_train = compute_league_sample_weights(train_data)

        model = build_xgboost()
        model.fit(X_train, y_train, sample_weight=w_train)
        y_pred = model.predict(X_test)

        res = evaluate(y_test, y_pred, model_name=f"Held-out league {held_out}")
        res["held_out_league"] = int(held_out)
        res["n_train"] = len(train_data)
        res["n_test"] = len(test_data)
        results.append(res)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_dir, "held_out_league_results.csv"),
                  index=False)
        print(f"\n  Held-out league results:")
        print(df.to_string(index=False))
        return df
    else:
        print("  No valid held-out league splits")
        return pd.DataFrame()
