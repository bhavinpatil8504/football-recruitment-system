"""
Stage 2: Feature Engineering
-----------------------------
Computes per-player-season features from raw StatsBomb event data:

  1. xG — logistic regression trained on shot events (transparent baseline)
  2. VAEP — Valuing Actions by Estimating Probabilities (Decroos et al., KDD 2019)
            Uses socceraction library: events → SPADL → game-state features →
            gradient-boosted scoring/conceding models → per-action value
  3. xT  — Expected Threat (Singh 2019): Markov-chain grid model that values
            ball movement by location change (16×12 pitch grid)
  4. Descriptive stats — pass completion, progressive passes, key passes,
            dribble success, pressure count
  5. Tactical phase breakdowns — all metrics split by pitch zone
            (build-up / progression / final third / box)

VAEP is the gold standard for action valuation in the literature and replaces
hand-weighted composites as the primary player-value signal. xT provides a
complementary position-based valuation (independent of action outcome).

Install: pip install socceraction statsbombpy scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

# ── VAEP / xT availability ──────────────────────────────────────────────────
try:
    import socceraction.spadl as spadl
    import socceraction.vaep.features as vaep_features
    import socceraction.vaep.labels as vaep_labels
    from socceraction.xthreat import ExpectedThreat
    SOCCERACTION_AVAILABLE = True
except ImportError:
    SOCCERACTION_AVAILABLE = False
    print("  socceraction not installed — VAEP/xT will be skipped.")
    print("  Install with: pip install socceraction==1.4.2")


# ── Tactical phases ───────────────────────────────────────────────────────────
# We tag every event with one of four tactical phases.
# This is a key novelty — most pipelines don't do this.
TACTICAL_PHASES = {
    "build_up":      {"min_x": 0,   "max_x": 40},   # own half, possession build
    "progression":   {"min_x": 40,  "max_x": 60},   # midfield transition zone
    "final_third":   {"min_x": 60,  "max_x": 80},   # attacking buildup
    "box":           {"min_x": 80,  "max_x": 120},  # penalty area threat
}


def tag_tactical_phase(events: pd.DataFrame) -> pd.DataFrame:
    """Add a 'tactical_phase' column based on event location."""
    events = events.copy()

    # StatsBomb location is [x, y] stored as a list
    def extract_x(loc):
        if isinstance(loc, list) and len(loc) >= 1:
            return loc[0]
        return np.nan

    events["loc_x"] = events["location"].apply(extract_x)

    conditions = [
        (events["loc_x"] >= 0)   & (events["loc_x"] < 40),
        (events["loc_x"] >= 40)  & (events["loc_x"] < 60),
        (events["loc_x"] >= 60)  & (events["loc_x"] < 80),
        (events["loc_x"] >= 80)  & (events["loc_x"] <= 120),
    ]
    choices = ["build_up", "progression", "final_third", "box"]
    events["tactical_phase"] = np.select(conditions, choices, default="progression")

    return events


# ── xG Model ──────────────────────────────────────────────────────────────────
class xGModel:
    """
    Simple logistic regression xG model.
    Features: distance to goal, angle to goal, body part, shot technique.
    
    Training requires shot events with 'shot_outcome' labels.
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=42))
        ])
        self.fitted = False

    def _extract_shot_features(self, shots: pd.DataFrame) -> pd.DataFrame:
        """Extract numerical features from shot events."""
        feats = pd.DataFrame()

        # Goal center in StatsBomb coordinate system
        GOAL_X, GOAL_Y = 120.0, 40.0

        def get_loc(row, idx):
            loc = row.get("location")
            if isinstance(loc, list) and len(loc) > idx:
                return loc[idx]
            return np.nan

        feats["x"] = shots.apply(lambda r: get_loc(r, 0), axis=1)
        feats["y"] = shots.apply(lambda r: get_loc(r, 1), axis=1)

        feats["distance"] = np.sqrt(
            (feats["x"] - GOAL_X)**2 + (feats["y"] - GOAL_Y)**2
        )
        # Angle in radians to the goal
        feats["angle"] = np.arctan2(
            np.abs(feats["y"] - GOAL_Y),
            np.maximum(GOAL_X - feats["x"], 0.1)
        )
        # Head shot indicator
        feats["is_header"] = shots.get("shot_body_part", pd.Series()).apply(
            lambda x: 1 if (isinstance(x, dict) and x.get("id") == 37)
                        or (isinstance(x, str) and x == "Head") else 0
        )
        # Big chance indicator (StatsBomb qualitative flag)
        feats["under_pressure"] = shots.get("under_pressure", pd.Series()).fillna(0).astype(int)

        median_vals = feats.median()
        if median_vals.isna().all():
            return feats.fillna(0)
        return feats.fillna(median_vals)

    def fit(self, shot_events: pd.DataFrame) -> "xGModel":
        """Train on shot events. Outcome 1 = goal."""
        shots = shot_events[shot_events["type"].apply(
            lambda t: (t.get("name", "") == "Shot" if isinstance(t, dict)
                       else str(t) == "Shot")
        )].copy()

        if len(shots) < 50:
            print("  Warning: fewer than 50 shots for xG training. Using dummy model.")
            self.fitted = False
            return self

        shots["is_goal"] = shots["shot_outcome"].apply(
            lambda x: 1 if (isinstance(x, dict) and x.get("id") == 98)
                        or (isinstance(x, str) and x == "Goal") else 0
        )

        X = self._extract_shot_features(shots)
        y = shots["is_goal"].values

        if y.sum() < 5:
            print("  Warning: too few goals in training data.")
            self.fitted = False
            return self

        self.model.fit(X, y)
        self.fitted = True
        print(f"  xG model trained on {len(shots)} shots "
              f"({y.sum()} goals, {y.mean():.3f} conversion rate)")
        return self

    def predict(self, shot_events: pd.DataFrame) -> pd.Series:
        """Return xG probability for each shot event."""
        shots = shot_events.copy()
        X = self._extract_shot_features(shots)
        if self.fitted:
            return pd.Series(self.model.predict_proba(X)[:, 1],
                             index=shots.index)
        else:
            # Naive fallback: distance-based
            GOAL_X, GOAL_Y = 120.0, 40.0
            dist = np.sqrt((X["x"] - GOAL_X)**2 + (X["y"] - GOAL_Y)**2)
            return pd.Series(np.exp(-dist / 25), index=shots.index)


# ── Per-player feature aggregation ────────────────────────────────────────────
def extract_player_features(events: pd.DataFrame,
                             xg_model: xGModel) -> pd.DataFrame:
    """
    For each player-season, compute:
      - xG, xG_against (conceded)
      - Pass completion rate, progressive passes, key passes
      - Pressure actions (pressing intensity)
      - Dribble success rate
      - Tactical phase breakdowns (each metric × 4 phases)
    
    Returns a wide DataFrame: one row per (player_id, season_id).
    """
    print("  Tagging tactical phases...")
    events = tag_tactical_phase(events)

    # ── Type helpers ──────────────────────────────────────────────────────────
    def type_name(t):
        if isinstance(t, dict):
            return t.get("name", "")
        return str(t)

    def outcome_name(o):
        if isinstance(o, dict):
            return o.get("name", "")
        return str(o) if pd.notna(o) else ""

    events["type_name"] = events["type"].apply(type_name)

    # ── Shot events → compute xG ──────────────────────────────────────────────
    shots = events[events["type_name"] == "Shot"].copy()
    if len(shots) > 0:
        shots["xg"] = xg_model.predict(shots)
    else:
        shots["xg"] = 0.0
    events.loc[shots.index, "xg"] = shots["xg"]

    # ── Pass features ─────────────────────────────────────────────────────────
    passes = events[events["type_name"] == "Pass"].copy()
    passes["pass_complete"] = passes["pass_outcome"].apply(
        lambda o: 1 if pd.isna(o) or outcome_name(o) == "" else 0
    )
    passes["key_pass"] = passes.get("pass_goal_assist", pd.Series(False)).fillna(False).astype(int)
    # Progressive pass: moves ball ≥ 10 yards toward opponent goal
    def is_progressive(row):
        loc  = row.get("location")
        end  = row.get("pass_end_location")
        if isinstance(loc, list) and isinstance(end, list) and len(loc) >= 1 and len(end) >= 1:
            return 1 if (end[0] - loc[0]) >= 10 else 0
        return 0
    passes["progressive"] = passes.apply(is_progressive, axis=1)

    # ── Dribble features ──────────────────────────────────────────────────────
    dribbles = events[events["type_name"] == "Dribble"].copy()
    dribbles["dribble_success"] = dribbles["dribble_outcome"].apply(
        lambda o: 1 if outcome_name(o) == "Complete" else 0
    )

    # ── Pressure / pressing ───────────────────────────────────────────────────
    pressures = events[events["type_name"] == "Pressure"].copy()

    # ── Aggregate per player per season ──────────────────────────────────────
    group_cols = ["player_id", "player", "season_id", "competition_id", "tactical_phase"]

    def safe_agg(df, group_cols, value_col, agg="sum"):
        if value_col not in df.columns or len(df) == 0:
            return pd.DataFrame(columns=group_cols + [value_col])
        return df.groupby(group_cols)[value_col].agg(agg).reset_index()

    # xG by phase
    xg_by_phase = (
        shots.groupby(["player_id", "player", "season_id", "competition_id", "tactical_phase"])
        ["xg"].sum().reset_index()
        .rename(columns={"xg": "xg_sum"})
    )

    # Pass metrics by phase
    pass_agg = (
        passes.groupby(["player_id", "player", "season_id", "competition_id", "tactical_phase"])
        .agg(
            passes_total    = ("pass_complete", "count"),
            passes_complete = ("pass_complete", "sum"),
            key_passes      = ("key_pass", "sum"),
            progressive_passes = ("progressive", "sum"),
        )
        .reset_index()
    )
    pass_agg["pass_completion_rate"] = (
        pass_agg["passes_complete"] / pass_agg["passes_total"].clip(lower=1)
    )

    # Dribble metrics by phase
    dribble_agg = (
        dribbles.groupby(["player_id", "player", "season_id", "competition_id", "tactical_phase"])
        .agg(
            dribbles_total   = ("dribble_success", "count"),
            dribbles_success = ("dribble_success", "sum"),
        )
        .reset_index()
    )
    dribble_agg["dribble_success_rate"] = (
        dribble_agg["dribbles_success"] / dribble_agg["dribbles_total"].clip(lower=1)
    )

    # Pressure count by phase
    pressure_agg = (
        pressures.groupby(["player_id", "player", "season_id", "competition_id", "tactical_phase"])
        .size().reset_index(name="pressure_count")
    )

    # ── Merge all phase-level features ───────────────────────────────────────
    merge_keys = ["player_id", "player", "season_id", "competition_id", "tactical_phase"]
    features = xg_by_phase
    for df in [pass_agg, dribble_agg, pressure_agg]:
        if len(df) > 0:
            features = features.merge(df, on=merge_keys, how="outer")
    features = features.fillna(0)

    # ── Pivot to wide format (phase × metric) ─────────────────────────────────
    # e.g.: xg_sum_box, xg_sum_final_third, pass_completion_rate_build_up ...
    id_cols    = ["player_id", "player", "season_id", "competition_id"]
    value_cols = [c for c in features.columns if c not in id_cols + ["tactical_phase"]]

    wide = features.pivot_table(
        index=id_cols,
        columns="tactical_phase",
        values=value_cols,
        aggfunc="sum",
        fill_value=0
    )
    wide.columns = ["_".join(str(c) for c in col) for col in wide.columns]
    wide = wide.reset_index()

    # Also keep overall (phase-agnostic) totals
    overall = features.groupby(id_cols)[value_cols].sum().reset_index()
    overall.columns = id_cols + [f"{c}_total" for c in value_cols]

    wide = wide.merge(overall, on=id_cols, how="left")

    print(f"  Feature matrix shape: {wide.shape}")
    return wide


# ══════════════════════════════════════════════════════════════════════════════
# VAEP + xT — Action-Level Valuation (socceraction)
# ══════════════════════════════════════════════════════════════════════════════

def _statsbomb_to_spadl(events: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw StatsBomb events (from statsbombpy) to SPADL format.

    socceraction.spadl.statsbomb.convert_to_actions() expects specific columns.
    statsbombpy returns a flat DataFrame with columns like 'type', 'location',
    'player', 'team', etc. We need to ensure compatibility.

    Returns a SPADL actions DataFrame with columns:
      game_id, period_id, time_seconds, team_id, player_id,
      start_x, start_y, end_x, end_y, type_id, result_id, bodypart_id
    """
    if not SOCCERACTION_AVAILABLE:
        raise ImportError("socceraction not installed")

    # socceraction expects the StatsBomb JSON structure. statsbombpy's sb.events()
    # returns a DataFrame that's close but needs minor adjustments.
    # We use socceraction's converter which handles the column mapping.
    try:
        # Group events by match and convert each match separately
        actions_list = []
        match_ids = events["match_id"].unique()

        for match_id in match_ids:
            match_events = events[events["match_id"] == match_id].copy()

            # socceraction needs 'id' column (StatsBomb event UUID)
            if "id" not in match_events.columns and "event_id" in match_events.columns:
                match_events["id"] = match_events["event_id"]

            # Determine home team for this match
            home_team = match_events["home_team"].iloc[0] if "home_team" in match_events.columns else None
            home_team_id = None
            if home_team and "team" in match_events.columns:
                # Find team_id for home team
                team_col = match_events["team"]
                if isinstance(team_col.iloc[0], dict):
                    home_mask = team_col.apply(lambda t: t.get("name", "") == home_team if isinstance(t, dict) else False)
                    if home_mask.any():
                        home_team_id = team_col[home_mask].iloc[0].get("id")

            try:
                match_actions = spadl.statsbomb.convert_to_actions(
                    match_events, home_team_id=home_team_id or 0
                )
                match_actions["game_id"] = match_id
                # Carry metadata
                match_actions["competition_id"] = match_events["competition_id"].iloc[0]
                match_actions["season_id"] = match_events["season_id"].iloc[0]
                actions_list.append(match_actions)
            except Exception as e:
                # Some matches may fail conversion — skip gracefully
                continue

        if not actions_list:
            raise ValueError("No matches could be converted to SPADL format")

        actions = pd.concat(actions_list, ignore_index=True)
        actions = spadl.add_names(actions)
        print(f"  SPADL conversion: {len(actions):,} actions from {len(actions_list)} matches")
        return actions

    except Exception as e:
        print(f"  SPADL conversion failed: {e}")
        raise


def compute_vaep(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VAEP (Valuing Actions by Estimating Probabilities) per action.

    VAEP (Decroos et al., KDD 2019) trains two gradient-boosted classifiers:
      P(scoring within 10 actions | game state after action a)
      P(conceding within 10 actions | game state after action a)

    Action value = ΔP(scoring) − ΔP(conceding)
      = [P(score|after) − P(score|before)] − [P(concede|after) − P(concede|before)]

    Returns DataFrame with columns:
      game_id, action_id, player_id, offensive_value, defensive_value, vaep_value
    """
    if not SOCCERACTION_AVAILABLE:
        print("  VAEP skipped — socceraction not installed")
        return pd.DataFrame()

    print("  Computing VAEP...")

    # Step 1: Convert to SPADL
    try:
        actions = _statsbomb_to_spadl(events)
    except Exception as e:
        print(f"  VAEP failed at SPADL conversion: {e}")
        return pd.DataFrame()

    # Step 2: Compute game states (sequences of 3 consecutive actions)
    print("  Building game states...")
    try:
        games = actions["game_id"].unique()
        all_features = []
        all_labels = []
        all_action_refs = []

        for game_id in games:
            game_actions = actions[actions["game_id"] == game_id].reset_index(drop=True)
            if len(game_actions) < 3:
                continue

            # Build game states (triplets of consecutive actions)
            gamestates = vaep_features.gamestates(game_actions, nb_prev_actions=3)

            # Compute features from game states
            X_game = pd.concat([
                vaep_features.actiontype(gamestates),
                vaep_features.result(gamestates),
                vaep_features.bodypart(gamestates),
                vaep_features.time(gamestates),
                vaep_features.startlocation(gamestates),
                vaep_features.endlocation(gamestates),
                vaep_features.movement(gamestates),
                vaep_features.space_delta(gamestates),
                vaep_features.team(gamestates),
                vaep_features.time_delta(gamestates),
            ], axis=1)

            # Compute labels (scoring/conceding within 10 actions)
            Y_game = pd.concat([
                vaep_labels.scores(game_actions, nr_actions=10),
                vaep_labels.concedes(game_actions, nr_actions=10),
            ], axis=1)

            all_features.append(X_game)
            all_labels.append(Y_game)
            all_action_refs.append(game_actions)

        if not all_features:
            print("  VAEP: no valid games for feature extraction")
            return pd.DataFrame()

        X = pd.concat(all_features, ignore_index=True)
        Y = pd.concat(all_labels, ignore_index=True)
        actions_concat = pd.concat(all_action_refs, ignore_index=True)

        print(f"  VAEP features: {X.shape[0]:,} actions × {X.shape[1]} features")

    except Exception as e:
        print(f"  VAEP feature extraction failed: {e}")
        return pd.DataFrame()

    # Step 3: Train scoring and conceding models
    print("  Training VAEP models (scoring + conceding)...")
    try:
        # Use gradient boosted classifiers (as in the original VAEP paper)
        model_scores = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_split=10, random_state=42
        )
        model_concedes = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_split=10, random_state=42
        )

        # Handle any remaining NaN
        X_clean = X.fillna(0)
        Y_scores = Y.iloc[:, 0].values   # scores column
        Y_concedes = Y.iloc[:, 1].values  # concedes column

        model_scores.fit(X_clean, Y_scores)
        model_concedes.fit(X_clean, Y_concedes)

        # Predict probabilities
        p_scores = model_scores.predict_proba(X_clean)[:, 1]
        p_concedes = model_concedes.predict_proba(X_clean)[:, 1]

        # VAEP value = ΔP(scoring) − ΔP(conceding) for each action
        # Offensive value = P(score|after) − P(score|before)
        # Defensive value = P(concede|before) − P(concede|after)
        offensive_value = np.zeros(len(p_scores))
        defensive_value = np.zeros(len(p_concedes))

        # Compute deltas within each game
        offset = 0
        for game_actions in all_action_refs:
            n = len(game_actions)
            game_p_scores = p_scores[offset:offset + n]
            game_p_concedes = p_concedes[offset:offset + n]

            # Offensive: how much does this action increase scoring probability?
            offensive_value[offset + 1:offset + n] = np.diff(game_p_scores)
            # Defensive: how much does this action decrease conceding probability?
            defensive_value[offset + 1:offset + n] = -np.diff(game_p_concedes)

            offset += n

        vaep_value = offensive_value + defensive_value

        # Build result DataFrame
        vaep_df = actions_concat[["game_id", "player_id", "competition_id", "season_id"]].copy()
        vaep_df["offensive_value"] = offensive_value
        vaep_df["defensive_value"] = defensive_value
        vaep_df["vaep_value"] = vaep_value

        print(f"  VAEP computed: mean={vaep_value.mean():.4f}, "
              f"std={vaep_value.std():.4f}, "
              f"range=[{vaep_value.min():.4f}, {vaep_value.max():.4f}]")

        return vaep_df

    except Exception as e:
        print(f"  VAEP model training failed: {e}")
        return pd.DataFrame()


def compute_xt(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Expected Threat (xT) per action using socceraction's
    Markov-chain grid model (Singh 2019).

    xT values ball-moving actions (passes, dribbles) by the change in
    scoring probability from the start location to the end location on
    a 16×12 pitch grid. Unlike VAEP, xT is purely location-based and
    does not consider action outcome — it values the attempt, not the result.

    Returns DataFrame with columns:
      game_id, player_id, competition_id, season_id, xt_value
    """
    if not SOCCERACTION_AVAILABLE:
        print("  xT skipped — socceraction not installed")
        return pd.DataFrame()

    print("  Computing xT (Expected Threat)...")

    try:
        actions = _statsbomb_to_spadl(events)

        # Train xT model on all actions
        xt_model = ExpectedThreat(l=16, w=12)
        xt_model.fit(actions)

        # Rate each action
        xt_values = xt_model.rate(actions)

        xt_df = actions[["game_id", "player_id", "competition_id", "season_id"]].copy()
        xt_df["xt_value"] = xt_values

        print(f"  xT computed: mean={xt_values.mean():.4f}, "
              f"std={xt_values.std():.4f}")

        return xt_df

    except Exception as e:
        print(f"  xT computation failed: {e}")
        return pd.DataFrame()


def aggregate_vaep_xt(vaep_df: pd.DataFrame,
                       xt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-action VAEP and xT values to per-player-season totals.

    Returns DataFrame with columns:
      player_id, season_id, competition_id,
      vaep_sum, vaep_offensive, vaep_defensive, vaep_per_action,
      xt_sum, xt_per_action, n_valued_actions
    """
    id_cols = ["player_id", "season_id", "competition_id"]
    result = pd.DataFrame()

    if len(vaep_df) > 0:
        vaep_agg = (
            vaep_df.groupby(id_cols)
            .agg(
                vaep_sum=("vaep_value", "sum"),
                vaep_offensive=("offensive_value", "sum"),
                vaep_defensive=("defensive_value", "sum"),
                vaep_per_action=("vaep_value", "mean"),
                n_valued_actions=("vaep_value", "count"),
            )
            .reset_index()
        )
        result = vaep_agg

    if len(xt_df) > 0:
        xt_agg = (
            xt_df.groupby(id_cols)
            .agg(
                xt_sum=("xt_value", "sum"),
                xt_per_action=("xt_value", "mean"),
            )
            .reset_index()
        )
        if len(result) > 0:
            result = result.merge(xt_agg, on=id_cols, how="outer")
        else:
            result = xt_agg

    if len(result) > 0:
        result = result.fillna(0)
        print(f"  VAEP/xT aggregated for {len(result)} player-seasons")
    else:
        print("  No VAEP/xT data to aggregate")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    print("Loading saved events...")
    events = pd.read_parquet("outputs/raw_events.parquet")

    print("Training xG model...")
    xg_model = xGModel()
    xg_model.fit(events)

    print("Extracting player features...")
    features = extract_player_features(events, xg_model)

    print(f"\nSample features:")
    print(features.head(3).T)

    features.to_parquet("outputs/player_features.parquet", index=False)
    print("\nSaved to outputs/player_features.parquet")
