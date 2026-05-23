"""
Tactical Blueprint — Style-Aware Recruitment
----------------------------------------------
KEY NOVELTY: Most recruitment analytics treat player value as absolute.
This module makes it SYSTEM-DEPENDENT — a player's value depends on
how well they fit the manager's desired tactical style.

Tactical Style Vector (8 dimensions):
    1. pressing_intensity    — how aggressively the team presses
    2. possession_preference — preference for retaining ball vs. direct play
    3. build_up_speed        — slow patient buildup vs. fast transitions
    4. width_of_play         — how much the team uses wide areas
    5. defensive_line_height — high line vs. deep block
    6. counter_attack_tendency — frequency of counter-attacking patterns
    7. direct_play_ratio     — long balls / bypassing midfield
    8. creative_freedom      — individual dribbles & risk-taking in final third

Each dimension is computed from event data and normalized to [0, 1].
Players get a tactical profile; managers specify a desired blueprint.
The ILP optimizer then maximizes TACTICAL FIT alongside raw impact.

Reference styles (presets):
    - Klopp (gegenpressing): high press, fast build-up, wide, high line
    - Guardiola (positional): high possession, patient build-up, high line
    - Simeone (low block):   low press, deep line, counter-attack focus
    - Mourinho (pragmatic):  medium press, direct play, deep line
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ── Preset tactical styles ────────────────────────────────────────────────────
PRESET_STYLES = {
    "gegenpressing": {
        "pressing_intensity":     0.95,
        "possession_preference":  0.60,
        "build_up_speed":         0.85,
        "width_of_play":          0.80,
        "defensive_line_height":  0.90,
        "counter_attack_tendency":0.40,
        "direct_play_ratio":      0.55,
        "creative_freedom":       0.65,
    },
    "positional_play": {
        "pressing_intensity":     0.70,
        "possession_preference":  0.95,
        "build_up_speed":         0.35,
        "width_of_play":          0.85,
        "defensive_line_height":  0.80,
        "counter_attack_tendency":0.15,
        "direct_play_ratio":      0.20,
        "creative_freedom":       0.50,
    },
    "low_block_counter": {
        "pressing_intensity":     0.20,
        "possession_preference":  0.30,
        "build_up_speed":         0.90,
        "width_of_play":          0.40,
        "defensive_line_height":  0.15,
        "counter_attack_tendency":0.90,
        "direct_play_ratio":      0.75,
        "creative_freedom":       0.30,
    },
    "pragmatic": {
        "pressing_intensity":     0.45,
        "possession_preference":  0.40,
        "build_up_speed":         0.70,
        "width_of_play":          0.50,
        "defensive_line_height":  0.35,
        "counter_attack_tendency":0.65,
        "direct_play_ratio":      0.60,
        "creative_freedom":       0.35,
    },
    "balanced": {
        "pressing_intensity":     0.50,
        "possession_preference":  0.50,
        "build_up_speed":         0.50,
        "width_of_play":          0.50,
        "defensive_line_height":  0.50,
        "counter_attack_tendency":0.50,
        "direct_play_ratio":      0.50,
        "creative_freedom":       0.50,
    },
}

TACTICAL_DIMS = list(PRESET_STYLES["balanced"].keys())


# ── Extract tactical profiles from event data ─────────────────────────────────
def extract_tactical_profiles(events: pd.DataFrame,
                               player_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an 8-dimensional tactical profile for each player
    based on their event data.

    Each dimension is derived from observable on-pitch actions:
      - pressing_intensity    = pressures per 90 minutes
      - possession_preference = pass completion rate × passes per 90
      - build_up_speed        = ratio of progressive passes to total passes
      - width_of_play         = fraction of actions in wide zones (y < 20 or y > 60)
      - defensive_line_height = average x-position of defensive actions
      - counter_attack_tendency = ratio of final_third actions after turnovers
      - direct_play_ratio     = long passes / total passes
      - creative_freedom      = (dribbles + key passes) in final third per 90

    Returns DataFrame with player_id + 8 tactical columns, all in [0, 1].
    """
    # Select only the columns we need to avoid copying the full 69-column DataFrame
    _needed_cols = [
        "player_id", "season_id", "type", "minute", "location",
        "pass_end_location", "pass_outcome", "pass_goal_assist",
        "pass_shot_assist", "carry_end_location",
    ]
    _needed_cols = [c for c in _needed_cols if c in events.columns]
    events = events[_needed_cols].copy()

    # Helper to get event type name
    def type_name(t):
        if isinstance(t, dict):
            return t.get("name", "")
        return str(t) if pd.notna(t) else ""

    events["_type_name"] = events["type"].apply(type_name)

    # Extract location components
    def get_x(loc):
        if isinstance(loc, list) and len(loc) >= 1:
            return float(loc[0])
        return np.nan

    def get_y(loc):
        if isinstance(loc, list) and len(loc) >= 2:
            return float(loc[1])
        return np.nan

    events["_loc_x"] = events["location"].apply(get_x)
    events["_loc_y"] = events["location"].apply(get_y)

    # Estimate minutes played per player per season (rough: events spread)
    player_minutes = (
        events.groupby(["player_id", "season_id"])["minute"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    player_minutes["est_minutes"] = (
        player_minutes["max"] - player_minutes["min"]
    ).clip(lower=45)  # minimum 45 min to avoid division issues
    player_minutes["per_90"] = 90.0 / player_minutes["est_minutes"]

    # Group key for per-player-season
    gk = ["player_id", "season_id"]

    # ── 1. PRESSING INTENSITY — pressures per 90 ────────────────────────────
    pressures = events[events["_type_name"] == "Pressure"]
    press_count = pressures.groupby(gk).size().reset_index(name="n_pressures")
    press_count = press_count.merge(
        player_minutes[["player_id", "season_id", "per_90"]], on=gk, how="left"
    )
    press_count["pressing_intensity"] = press_count["n_pressures"] * press_count["per_90"].fillna(1)

    # ── 2. POSSESSION PREFERENCE — pass completion × volume ─────────────────
    passes = events.loc[events["_type_name"] == "Pass"].copy()

    def outcome_name(o):
        if isinstance(o, dict):
            return o.get("name", "")
        return str(o) if pd.notna(o) else ""

    passes["_complete"] = passes["pass_outcome"].apply(
        lambda o: 1 if pd.isna(o) or outcome_name(o) == "" else 0
    )
    pass_stats = passes.groupby(gk).agg(
        total_passes=("_complete", "count"),
        completed=("_complete", "sum"),
    ).reset_index()
    pass_stats = pass_stats.merge(
        player_minutes[["player_id", "season_id", "per_90"]], on=gk, how="left"
    )
    pass_stats["completion_rate"] = pass_stats["completed"] / pass_stats["total_passes"].clip(lower=1)
    pass_stats["passes_per_90"] = pass_stats["total_passes"] * pass_stats["per_90"].fillna(1)
    pass_stats["possession_preference"] = (
        pass_stats["completion_rate"] * pass_stats["passes_per_90"]
    )

    # ── 3. BUILD-UP SPEED — progressive pass ratio ──────────────────────────
    def is_progressive(row):
        loc = row.get("location")
        end = row.get("pass_end_location")
        if isinstance(loc, list) and isinstance(end, list) and len(loc) >= 1 and len(end) >= 1:
            return 1 if (end[0] - loc[0]) >= 10 else 0
        return 0

    passes["_progressive"] = passes.apply(is_progressive, axis=1)
    prog_stats = passes.groupby(gk).agg(
        total=("_progressive", "count"),
        progressive=("_progressive", "sum"),
    ).reset_index()
    prog_stats["build_up_speed"] = prog_stats["progressive"] / prog_stats["total"].clip(lower=1)

    # ── 4. WIDTH OF PLAY — fraction of actions in wide zones ────────────────
    wide_events = events.dropna(subset=["_loc_y"])
    wide_events["_is_wide"] = ((wide_events["_loc_y"] < 20) | (wide_events["_loc_y"] > 60)).astype(int)
    width_stats = wide_events.groupby(gk).agg(
        total_actions=("_is_wide", "count"),
        wide_actions=("_is_wide", "sum"),
    ).reset_index()
    width_stats["width_of_play"] = width_stats["wide_actions"] / width_stats["total_actions"].clip(lower=1)

    # ── 5. DEFENSIVE LINE HEIGHT — avg x-position of defensive actions ──────
    defensive_types = {"Block", "Interception", "Clearance", "Pressure", "Tackle"}
    def_events = events[events["_type_name"].isin(defensive_types)].dropna(subset=["_loc_x"])
    def_height = def_events.groupby(gk)["_loc_x"].mean().reset_index()
    def_height.columns = gk + ["defensive_line_height"]

    # ── 6. COUNTER-ATTACK TENDENCY ──────────────────────────────────────────
    # Proxy: ratio of carries/passes that START in own half and END in final third
    carries_passes = events.loc[events["_type_name"].isin({"Pass", "Carry"})].copy()

    def get_end_x(row):
        for col in ["pass_end_location", "carry_end_location"]:
            val = row.get(col) if col in row.index else None
            if isinstance(val, list) and len(val) >= 1:
                return float(val[0])
        return np.nan

    carries_passes["_end_x"] = carries_passes.apply(get_end_x, axis=1)
    carries_passes["_is_counter"] = (
        (carries_passes["_loc_x"] < 50) & (carries_passes["_end_x"] >= 80)
    ).astype(int)
    counter_stats = carries_passes.groupby(gk).agg(
        total=("_is_counter", "count"),
        counters=("_is_counter", "sum"),
    ).reset_index()
    counter_stats["counter_attack_tendency"] = (
        counter_stats["counters"] / counter_stats["total"].clip(lower=1)
    )

    # ── 7. DIRECT PLAY RATIO — long passes / total passes ───────────────────
    def pass_length(row):
        loc = row.get("location")
        end = row.get("pass_end_location")
        if isinstance(loc, list) and isinstance(end, list) and len(loc) >= 2 and len(end) >= 2:
            return np.sqrt((end[0]-loc[0])**2 + (end[1]-loc[1])**2)
        return 0

    passes["_length"] = passes.apply(pass_length, axis=1)
    passes["_is_long"] = (passes["_length"] > 32).astype(int)  # >32 yards = long ball
    direct_stats = passes.groupby(gk).agg(
        total=("_is_long", "count"),
        long_passes=("_is_long", "sum"),
    ).reset_index()
    direct_stats["direct_play_ratio"] = direct_stats["long_passes"] / direct_stats["total"].clip(lower=1)

    # ── 8. CREATIVE FREEDOM — (dribbles + key passes) in final third per 90
    creative_types = {"Dribble"}
    creative_events = events[
        (events["_type_name"].isin(creative_types)) & (events["_loc_x"] >= 80)
    ]
    # Also count key passes in final third
    key_passes_ft = passes[
        (passes["_loc_x"] >= 80) &
        (passes.get("pass_goal_assist", pd.Series(False)).fillna(False).astype(bool) |
         passes.get("pass_shot_assist", pd.Series(False)).fillna(False).astype(bool))
    ] if "pass_goal_assist" in passes.columns or "pass_shot_assist" in passes.columns else pd.DataFrame()

    creative_count = creative_events.groupby(gk).size().reset_index(name="n_creative")
    if len(key_passes_ft) > 0:
        kp_count = key_passes_ft.groupby(gk).size().reset_index(name="n_key_ft")
        creative_count = creative_count.merge(kp_count, on=gk, how="outer").fillna(0)
        creative_count["n_creative"] = creative_count["n_creative"] + creative_count["n_key_ft"]

    creative_count = creative_count.merge(
        player_minutes[["player_id", "season_id", "per_90"]], on=gk, how="left"
    )
    creative_count["creative_freedom"] = creative_count["n_creative"] * creative_count["per_90"].fillna(1)

    # ── Merge all dimensions ─────────────────────────────────────────────────
    # Start from player list
    player_list = events[["player_id", "season_id"]].drop_duplicates()

    dim_dfs = [
        (press_count,   ["pressing_intensity"]),
        (pass_stats,    ["possession_preference"]),
        (prog_stats,    ["build_up_speed"]),
        (width_stats,   ["width_of_play"]),
        (def_height,    ["defensive_line_height"]),
        (counter_stats, ["counter_attack_tendency"]),
        (direct_stats,  ["direct_play_ratio"]),
        (creative_count,["creative_freedom"]),
    ]

    profiles = player_list.copy()
    for dim_df, cols in dim_dfs:
        profiles = profiles.merge(dim_df[gk + cols], on=gk, how="left")

    profiles = profiles.fillna(0)

    # ── Normalize each dimension to [0, 1] using RANK-BASED normalization ──
    # MinMaxScaler is distorted by outliers (e.g., one player with 50 pressures
    # per 90 compresses everyone else near zero). Rank-based normalization
    # (percentile transform) spreads players evenly across [0, 1].
    for dim in TACTICAL_DIMS:
        vals = profiles[dim]
        # Rank transform: ties get average rank, then scale to [0, 1]
        ranked = vals.rank(method="average", pct=True)
        profiles[dim] = ranked

    # ── Average across seasons for a stable profile ──────────────────────────
    player_profiles = profiles.groupby("player_id")[TACTICAL_DIMS].mean().reset_index()

    print(f"  Tactical profiles computed for {len(player_profiles)} players")
    print(f"  Dimensions: {TACTICAL_DIMS}")

    return player_profiles


# ── Tactical fit scoring ──────────────────────────────────────────────────────
def compute_tactical_fit(player_profiles: pd.DataFrame,
                          target_style: dict,
                          position_weights: dict = None) -> pd.DataFrame:
    """
    Score each player's FIT to a target tactical style.

    Fit = 1 - normalized_distance(player_profile, target_style)

    Higher fit → player's natural style matches the desired system.
    This is the key insight: a great player in the wrong system
    underperforms a good player in the right system.

    Parameters
    ----------
    player_profiles : DataFrame with player_id + tactical dimensions
    target_style    : dict mapping tactical dimension → desired value [0,1]
    position_weights: optional dict mapping position → dimension weights
                      (e.g., pressing_intensity matters more for ATT than GK)

    Returns
    -------
    DataFrame with player_id, tactical_fit (0-1), per-dimension distances
    """
    target = np.array([target_style.get(d, 0.5) for d in TACTICAL_DIMS])

    profiles_matrix = player_profiles[TACTICAL_DIMS].values  # (n_players, 8)

    # Euclidean distance, normalized to [0, 1]
    distances = np.sqrt(np.sum((profiles_matrix - target) ** 2, axis=1))
    max_possible = np.sqrt(len(TACTICAL_DIMS))  # worst case: all dims differ by 1
    normalized_dist = distances / max_possible

    fit_scores = 1.0 - normalized_dist

    result = player_profiles[["player_id"]].copy()
    result["tactical_fit"] = fit_scores

    # Per-dimension breakdown (useful for interpretation)
    for i, dim in enumerate(TACTICAL_DIMS):
        result[f"fit_{dim}"] = 1.0 - abs(profiles_matrix[:, i] - target[i])

    print(f"  Tactical fit computed for style: "
          f"mean={fit_scores.mean():.3f}, max={fit_scores.max():.3f}")

    return result


# ── Position-aware tactical weights ───────────────────────────────────────────
# Different dimensions matter more for different positions
POSITION_TACTICAL_WEIGHTS = {
    "GK":  {"pressing_intensity": 0.3, "possession_preference": 0.5,
             "build_up_speed": 0.8, "width_of_play": 0.1,
             "defensive_line_height": 0.9, "counter_attack_tendency": 0.2,
             "direct_play_ratio": 0.7, "creative_freedom": 0.1},
    "DEF": {"pressing_intensity": 0.7, "possession_preference": 0.6,
             "build_up_speed": 0.5, "width_of_play": 0.6,
             "defensive_line_height": 1.0, "counter_attack_tendency": 0.4,
             "direct_play_ratio": 0.5, "creative_freedom": 0.2},
    "MID": {"pressing_intensity": 0.8, "possession_preference": 0.9,
             "build_up_speed": 0.7, "width_of_play": 0.7,
             "defensive_line_height": 0.5, "counter_attack_tendency": 0.6,
             "direct_play_ratio": 0.6, "creative_freedom": 0.7},
    "ATT": {"pressing_intensity": 0.9, "possession_preference": 0.5,
             "build_up_speed": 0.6, "width_of_play": 0.8,
             "defensive_line_height": 0.3, "counter_attack_tendency": 0.9,
             "direct_play_ratio": 0.4, "creative_freedom": 1.0},
}


def compute_position_aware_fit(player_profiles: pd.DataFrame,
                                target_style: dict,
                                positions: pd.Series) -> pd.DataFrame:
    """
    Like compute_tactical_fit but weights dimensions by position.

    A pressing-heavy style weights pressing_intensity more for
    attackers (who initiate the press) than goalkeepers.
    """
    target = np.array([target_style.get(d, 0.5) for d in TACTICAL_DIMS])
    profiles_matrix = player_profiles[TACTICAL_DIMS].values

    fit_scores = []
    for i in range(len(player_profiles)):
        pos = positions.iloc[i] if i < len(positions) else "MID"
        weights = POSITION_TACTICAL_WEIGHTS.get(pos, POSITION_TACTICAL_WEIGHTS["MID"])
        w = np.array([weights.get(d, 0.5) for d in TACTICAL_DIMS])

        # Weighted distance
        diff = (profiles_matrix[i] - target) * w
        dist = np.sqrt(np.sum(diff ** 2)) / np.sqrt(np.sum(w ** 2))
        fit_scores.append(1.0 - dist)

    result = player_profiles[["player_id"]].copy()
    result["tactical_fit"] = fit_scores

    return result


# ── Modified ILP with tactical fit ────────────────────────────────────────────
def compute_style_adjusted_value(candidates: pd.DataFrame,
                                  tactical_fit: pd.DataFrame,
                                  alpha: float = 0.4) -> pd.DataFrame:
    """
    Blend raw impact with tactical fit using an ADDITIVE approach:

        V_style = V_adj + α · (tactical_fit - 0.5) · spread

    where spread = IQR of v_adj (robust scale of the value distribution).

    Why additive, not multiplicative?
      The old formula V_adj × (1 + α·(fit-0.5)) INVERTS the effect
      when V_adj < 0: a high-fit player with negative V_adj gets an
      even MORE negative V_style. With risk adjustment (λ > 0), the
      majority of players have V_adj < 0, breaking the entire mechanism.

    Additive blending guarantees:
      - α = 0.0 → V_style = V_adj (pure impact, no tactical influence)
      - α > 0   → high-fit players get a BONUS, low-fit get a PENALTY,
                   regardless of the sign of V_adj
      - The bonus is scaled to the IQR of V_adj so it's large enough
        to change rankings but not so large that a low-impact player
        with perfect fit overtakes a genuinely impactful one
      - (fit - 0.5) centering: average-fit players are unaffected

    Parameters
    ----------
    alpha : float in [0, 1]. At α=0 tactical fit is ignored.
            At α=1 a perfect-fit player gets +0.5·spread bonus.
    """
    df = candidates.merge(tactical_fit[["player_id", "tactical_fit"]],
                          on="player_id", how="left")
    df["tactical_fit"] = df["tactical_fit"].fillna(0.5)

    # Scale the bonus off y_hat (pre-risk predicted impact), NOT v_adj.
    # Why: v_adj is compressed by the risk penalty (mean ~ -0.10),
    # making its IQR tiny (~0.015). Using y_hat's spread ensures the
    # tactical bonus is large enough to actually change squad rankings.
    if "y_hat" in df.columns:
        spread = max(df["y_hat"].std(), 0.01)
    else:
        spread = max(df["v_adj"].std(), 0.01)

    # Additive blending: tactical fit adds/subtracts a meaningful bonus
    # At α=0.4, a perfect-fit player (fit=1.0) gets + 0.4·0.5·spread bonus
    # At α=0.4, a zero-fit player (fit=0.0) gets  - 0.4·0.5·spread penalty
    tactical_bonus = alpha * (df["tactical_fit"] - 0.5) * spread
    df["v_style"] = df["v_adj"] + tactical_bonus

    print(f"  Style-adjusted value: α={alpha}")
    print(f"  Mean v_adj       : {df['v_adj'].mean():.4f}")
    print(f"  Mean tactical_fit: {df['tactical_fit'].mean():.4f}")
    print(f"  Mean v_style     : {df['v_style'].mean():.4f}")
    print(f"  Spread (y_hat σ) : {spread:.4f}")
    print(f"  Tactical bonus range: [{tactical_bonus.min():.4f}, {tactical_bonus.max():.4f}]")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# LEARNED INTERACTION MODEL — upgrade from additive heuristic
# ══════════════════════════════════════════════════════════════════════════════

def train_style_interaction_model(transfers: pd.DataFrame,
                                   player_profiles: pd.DataFrame,
                                   features: pd.DataFrame,
                                   target_col: str = "impact_score") -> dict:
    """
    Train a model predicting post-transfer performance from
    player features × system features (style-fit interaction).

    This replaces the additive heuristic V_style = V_adj + α·(fit-0.5)·spread
    with a LEARNED interaction: the model discovers which player-style
    combinations produce gains vs. losses after a transfer.

    Model: GradientBoostingRegressor on features:
      - Player's 8-D tactical profile (pre-transfer)
      - Destination club's 8-D style proxy (computed from post-transfer events)
      - Player × style interaction terms (element-wise product)
      - Pre-transfer impact_score

    Target: post-transfer impact_score

    Returns dict with 'model', 'feature_names', 'metrics', 'n_transfers'.
    Returns None if insufficient transfer data (<20 transfers).
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from scipy.stats import spearmanr

    if transfers is None or len(transfers) < 20:
        print("  Interaction model: insufficient transfers (<20), "
              "falling back to heuristic")
        return None

    # Build training data: for each transfer, get player profile + destination style
    train_rows = []
    for _, t in transfers.iterrows():
        pid = t["player_id"]
        s_from = t["season_from"]
        s_to = t["season_to"]

        # Player's tactical profile (pre-transfer)
        p_row = player_profiles[player_profiles["player_id"] == pid]
        if len(p_row) == 0:
            continue
        player_style = p_row[TACTICAL_DIMS].values[0]

        # Pre-transfer impact
        pre_mask = (features["player_id"] == pid) & (features["season_id"] == s_from)
        pre_data = features[pre_mask]
        if len(pre_data) == 0 or target_col not in pre_data.columns:
            continue
        pre_impact = float(pre_data[target_col].iloc[0])

        # Post-transfer impact (target)
        post_mask = (features["player_id"] == pid) & (features["season_id"] == s_to)
        post_data = features[post_mask]
        if len(post_data) == 0 or target_col not in post_data.columns:
            continue
        post_impact = float(post_data[target_col].iloc[0])

        # Destination club style proxy: average tactical profile of all players
        # at the destination club in the post-transfer season
        club_to = t.get("club_to")
        if club_to and "team_name" in features.columns:
            club_mask = (features["team_name"] == club_to) & (features["season_id"] == s_to)
            club_pids = features[club_mask]["player_id"].unique()
            club_profiles = player_profiles[player_profiles["player_id"].isin(club_pids)]
            if len(club_profiles) >= 3:
                dest_style = club_profiles[TACTICAL_DIMS].mean().values
            else:
                dest_style = np.full(len(TACTICAL_DIMS), 0.5)
        else:
            dest_style = np.full(len(TACTICAL_DIMS), 0.5)

        # Interaction terms: element-wise product of player × destination
        interaction = player_style * dest_style

        row = list(player_style) + list(dest_style) + list(interaction) + [pre_impact]
        train_rows.append((row, post_impact))

    if len(train_rows) < 20:
        print(f"  Interaction model: only {len(train_rows)} valid transfers, "
              "need 20+. Falling back to heuristic.")
        return None

    X = np.array([r[0] for r in train_rows])
    y = np.array([r[1] for r in train_rows])

    # Feature names for interpretability
    feat_names = (
        [f"player_{d}" for d in TACTICAL_DIMS] +
        [f"dest_{d}" for d in TACTICAL_DIMS] +
        [f"interact_{d}" for d in TACTICAL_DIMS] +
        ["pre_impact"]
    )

    # Train with cross-validation
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_split=5, random_state=42
    )

    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 5),
                                 scoring="r2")
    model.fit(X, y)
    y_pred = model.predict(X)
    rho, _ = spearmanr(y, y_pred)

    metrics = {
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "train_spearman": float(rho),
        "n_transfers": len(train_rows),
    }

    print(f"  Interaction model trained on {len(train_rows)} transfers")
    print(f"  CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Train Spearman ρ = {rho:.3f}")

    # Feature importance
    importances = model.feature_importances_
    top_5 = np.argsort(importances)[-5:][::-1]
    print(f"  Top features: {[feat_names[i] for i in top_5]}")

    return {
        "model": model,
        "feature_names": feat_names,
        "metrics": metrics,
        "n_transfers": len(train_rows),
    }


def predict_system_fit(interaction_model: dict,
                        player_profiles: pd.DataFrame,
                        target_style: dict,
                        pre_impact: pd.Series) -> pd.Series:
    """
    Use the learned interaction model to predict how each player
    would perform in the target tactical system.

    Unlike the additive heuristic, this captures nonlinear interactions
    (e.g., a pressing-heavy attacker might thrive in gegenpressing but
    struggle in a low-block system, in ways the heuristic can't model).
    """
    model = interaction_model["model"]

    dest_style = np.array([target_style.get(d, 0.5) for d in TACTICAL_DIMS])

    X_rows = []
    for i in range(len(player_profiles)):
        player_style = player_profiles[TACTICAL_DIMS].values[i]
        interaction = player_style * dest_style
        row = list(player_style) + list(dest_style) + list(interaction) + [float(pre_impact.iloc[i])]
        X_rows.append(row)

    X = np.array(X_rows)
    predicted_post = model.predict(X)

    return pd.Series(predicted_post, index=player_profiles.index)


def compute_style_adjusted_value_learned(candidates: pd.DataFrame,
                                          interaction_model: dict,
                                          player_profiles: pd.DataFrame,
                                          target_style: dict,
                                          alpha: float = 0.4) -> pd.DataFrame:
    """
    Style-adjusted value using the LEARNED interaction model.

    V_style = (1-α)·V_adj + α·predicted_post_transfer

    When interaction_model is None, falls back to additive heuristic.
    """
    if interaction_model is None:
        print("  No interaction model — using additive heuristic")
        from src.tactics.blueprint import compute_tactical_fit
        tactical_fit = compute_tactical_fit(player_profiles, target_style)
        return compute_style_adjusted_value(candidates, tactical_fit, alpha)

    df = candidates.copy()

    # Get player profiles for candidates
    merged_profiles = player_profiles[
        player_profiles["player_id"].isin(df["player_id"])
    ].copy()

    # Align with candidates order
    merged_profiles = df[["player_id"]].merge(merged_profiles, on="player_id", how="left")
    for d in TACTICAL_DIMS:
        merged_profiles[d] = merged_profiles[d].fillna(0.5)

    # Get pre-impact
    pre_impact = df["v_adj"] if "v_adj" in df.columns else df.get("y_hat", pd.Series(0.5, index=df.index))

    # Predict post-transfer performance
    predicted_post = predict_system_fit(
        interaction_model, merged_profiles, target_style, pre_impact
    )

    # Blend: convex combination of risk-adjusted value and predicted system performance
    df["predicted_system_impact"] = predicted_post.values
    df["v_style"] = (1 - alpha) * df["v_adj"] + alpha * predicted_post.values

    # Also compute heuristic fit for comparison/reporting
    from src.tactics.blueprint import compute_tactical_fit
    tactical_fit = compute_tactical_fit(player_profiles, target_style)
    df = df.merge(tactical_fit[["player_id", "tactical_fit"]], on="player_id", how="left")
    df["tactical_fit"] = df["tactical_fit"].fillna(0.5)

    print(f"  Learned style-adjusted value: α={alpha}")
    print(f"  Mean V_adj              : {df['v_adj'].mean():.4f}")
    print(f"  Mean predicted_system   : {predicted_post.mean():.4f}")
    print(f"  Mean V_style (blended)  : {df['v_style'].mean():.4f}")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    events   = pd.read_parquet("outputs/raw_events.parquet")
    features = pd.read_parquet("outputs/player_features_normalized.parquet")

    # Extract tactical profiles
    profiles = extract_tactical_profiles(events, features)
    profiles.to_parquet("outputs/tactical_profiles.parquet", index=False)

    # Compute fit for different styles
    for style_name, style_vec in PRESET_STYLES.items():
        fit = compute_tactical_fit(profiles, style_vec)
        print(f"\n  {style_name}: mean_fit={fit['tactical_fit'].mean():.3f}, "
              f"max_fit={fit['tactical_fit'].max():.3f}")
