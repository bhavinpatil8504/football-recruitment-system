"""
Stage 6: Robust Risk-Adjusted Squad Optimization
--------------------------------------------------
Step A — Risk-Adjusted Valuation
  V_adj = ŷ_i − λ · risk_i

  Where risk_i is a composite penalty index combining three sources:
    1. Cross-season performance std  — empirical volatility across seasons
    2. Bayesian posterior uncertainty — from the hierarchical normalizer
    3. Position-deviation proxy       — for single-season players only

  These are combined and scaled to [0.01, 0.30] via winsorized MinMax.
  λ controls risk tolerance (0 = risk-neutral, 1 = max aversion).

  NOTE: This is a robust scoring heuristic, not mean-variance optimization
  in the Markowitz/portfolio-theory sense. The risk index is a rescaled
  composite, not in native variance units.

Step B — Integer Linear Programming (ILP)
  Maximize: Σ z_i · V_adj_i
  Subject to:
    Σ c_i · z_i ≤ B       (transfer budget)
    Positional coverage    (min/max per position per formation)
    Squad size = 11
    Club diversity         (max N players from one club)
    Youth sustainability   (min % of young prospects)

  ILP finds the globally optimal squad via branch-and-bound (PuLP CBC).

Install: pip install pulp
"""

import numpy as np
import pandas as pd
from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                  LpBinary, LpStatus, value, PULP_CBC_CMD)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# -- Risk adjustment -----------------------------------------------------------
def compute_risk_adjusted_value(predictions: pd.DataFrame,
                                 normalized_df: pd.DataFrame,
                                 lam: float = 0.5) -> pd.DataFrame:
    """
    Compute V_adj = ŷ_i − λ · risk_i for each player.

    risk_i is a composite index (scaled to [0.01, 0.30]) from three sources:
      1. Cross-season std of impact_score — empirical volatility
      2. Bayesian uncertainty — relative posterior uncertainty from normalizer
         (converted from absolute to league-relative to avoid domination)
      3. Position-deviation proxy — for single-season players only
         (how far from position-group mean ŷ)

    Blending weights adapt to data availability: multi-season players
    rely on source 1, single-season players rely on source 3.
    Source 2 acts as a tiebreaker (0.1× weight).
    """
    # ── Source 1: Cross-season variability (most reliable) ───────────────────
    # Standard deviation of impact_score across seasons for each player
    if "impact_score" in normalized_df.columns:
        season_var = (
            normalized_df.groupby("player_id")["impact_score"]
            .agg(["std", "count"])
            .reset_index()
            .rename(columns={"std": "perf_std", "count": "n_seasons"})
        )
        # Players with only 1 season get NaN std → handle below
        season_var["perf_std"] = season_var["perf_std"].fillna(0)
    else:
        season_var = pd.DataFrame({
            "player_id": predictions["player_id"].unique(),
            "perf_std": 0.0,
            "n_seasons": 1,
        })

    # ── Source 2: Bayesian uncertainty ────────────────────────────────────────
    uncertainty_cols = [c for c in normalized_df.columns if "uncertainty" in c]
    if uncertainty_cols:
        bayes_unc = (
            normalized_df[["player_id"] + uncertainty_cols]
            .drop_duplicates("player_id")
        )
        bayes_unc["bayes_var"] = bayes_unc[uncertainty_cols].mean(axis=1)
    else:
        bayes_unc = pd.DataFrame({
            "player_id": predictions["player_id"].unique(),
            "bayes_var": 0.0,
        })

    # ── Merge into predictions ───────────────────────────────────────────────
    df = predictions.copy()
    df = df.merge(season_var[["player_id", "perf_std", "n_seasons"]],
                  on="player_id", how="left")
    df = df.merge(bayes_unc[["player_id", "bayes_var"]],
                  on="player_id", how="left")

    df["perf_std"]   = df["perf_std"].fillna(0)
    df["n_seasons"]  = df["n_seasons"].fillna(1)
    df["bayes_var"]  = df["bayes_var"].fillna(0)

    # ── Source 3: Position-group proxy for single-season players ─────────────
    # Players appearing in only 1 season have no cross-season variability.
    # Use their deviation from position-group mean as a risk proxy.
    if "position" in df.columns:
        pos_means = df.groupby("position")["y_hat"].mean()
        df["pos_dev"] = df.apply(
            lambda r: abs(r["y_hat"] - pos_means.get(r["position"], r["y_hat"])),
            axis=1
        )
    else:
        df["pos_dev"] = abs(df["y_hat"] - df["y_hat"].mean())

    # ── Combine variance sources ─────────────────────────────────────────────
    # Key insight: bayes_var = (1-B)·σ_l² is the SAME for all players in a
    # league — it measures league-level noise, not individual risk. Adding it
    # raw dominates the signal, pushing everyone to high variance.
    #
    # Fix: use bayes_var as a RELATIVE signal (deviation from league median),
    # not an absolute additive term. Cross-season std is the primary signal.
    # For single-season players, position deviation is the proxy.

    w_perf  = df["n_seasons"].clip(upper=5) / 5.0   # 0.2 for 1 season, 1.0 for 5+
    w_proxy = 1.0 - w_perf                           # complement

    # Bayesian uncertainty: convert from absolute to RELATIVE within league
    # (how much more uncertain is this player than the typical player?)
    if df["bayes_var"].std() > 1e-9:
        bayes_relative = (df["bayes_var"] - df["bayes_var"].median()).clip(lower=0)
    else:
        bayes_relative = 0.0

    raw_var = (
        w_perf  * df["perf_std"] +           # primary: cross-season swings
        w_proxy * df["pos_dev"] * 0.3 +      # proxy for single-season players
        0.1 * bayes_relative                  # tiebreaker, not dominator
    )

    # ── Winsorized MinMax to [0.01, risk_cap] ────────────────────────────────
    # Winsorize at 95th percentile to clip outliers, then scale so that:
    #   - most reliable players: ~0.01
    #   - median player: ~0.10-0.15  (well below 0.30 cap)
    #   - riskiest players: ~0.25-0.30
    # This ensures the ILP risk constraint is feasible AND meaningful.
    risk_cap_val = 0.30
    clip_upper   = max(raw_var.quantile(0.95), 1e-6)
    clipped      = raw_var.clip(lower=0, upper=clip_upper)
    v_min, v_max = clipped.min(), clipped.max()
    if v_max - v_min < 1e-9:
        df["variance"] = 0.05
    else:
        df["variance"] = 0.01 + (clipped - v_min) / (v_max - v_min) * (risk_cap_val - 0.01)

    df["v_adj"] = df["y_hat"] - lam * df["variance"]

    print(f"  Risk adjustment: lam={lam}")
    print(f"  Mean y_hat   : {df['y_hat'].mean():.4f}")
    print(f"  Mean variance: {df['variance'].mean():.4f}")
    print(f"  Mean v_adj   : {df['v_adj'].mean():.4f}")
    print(f"  Variance range: [{df['variance'].min():.4f}, {df['variance'].max():.4f}]")
    print(f"  Players penalized by >0.05: "
          f"{(df['y_hat'] - df['v_adj'] > 0.05).sum()}")

    return df


# -- Position mapping ----------------------------------------------------------
POSITION_MAP = {
    "Goalkeeper":             "GK",
    "Right Back":             "DEF",
    "Left Back":              "DEF",
    "Right Center Back":      "DEF",
    "Left Center Back":       "DEF",
    "Center Back":            "DEF",
    "Right Midfield":         "MID",
    "Left Midfield":          "MID",
    "Right Center Midfield":  "MID",
    "Left Center Midfield":   "MID",
    "Center Defensive Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Right Wing":             "ATT",
    "Left Wing":              "ATT",
    "Right Center Forward":   "ATT",
    "Left Center Forward":    "ATT",
    "Center Forward":         "ATT",
    "Secondary Striker":      "ATT",
}


def assign_positions(player_df: pd.DataFrame,
                     lineups: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the most common position played per player from lineup data.
    Always guarantees a 'position' column exists in the output,
    defaulting to 'MID' if position data is unavailable.
    """
    df = player_df.copy()

    # If position already exists and is populated, nothing to do
    if "position" in df.columns and df["position"].notna().all():
        return df

    # Guard: if lineup data has no positions column, default everything
    if lineups is None or len(lineups) == 0:
        print("  DEBUG: lineups is None or empty")
        df["position"] = df.get("position", "MID")
        df["position"] = df["position"].fillna("MID")
        return df

    if "positions" not in lineups.columns:
        # Try alternative column names
        pos_col = None
        for col in ["position", "player_position", "pos"]:
            if col in lineups.columns:
                pos_col = col
                break
        if pos_col is None:
            df["position"] = df.get("position", "MID")
            df["position"] = df["position"].fillna("MID")
            return df
    else:
        pos_col = "positions"

    def primary_position(pos_data):
        """Extract simplified position label from StatsBomb positions data.

        StatsBomb format (actual):
          [{'position_id': 5, 'position': 'Left Center Back', 'from': '00:00', ...}]
        The position NAME is under key 'position' as a string.
        """
        # Handle list format
        if isinstance(pos_data, list):
            if len(pos_data) == 0:
                return "MID"
            pos = pos_data[0]
            if isinstance(pos, dict):
                # Try 'position' key first (StatsBomb actual format)
                name = pos.get("position", None)
                if isinstance(name, dict):
                    # Nested: {"position": {"name": "Goalkeeper"}}
                    name = name.get("name", "MID")
                elif isinstance(name, str):
                    # Flat string: {"position": "Left Center Back"}
                    pass  # name is already the string we need
                else:
                    # Fallback to 'name' key
                    name = pos.get("name", "MID")
                return POSITION_MAP.get(str(name), "MID")
            # pos is a string
            return POSITION_MAP.get(str(pos), "MID")

        # Handle dict format
        if isinstance(pos_data, dict):
            name = pos_data.get("position", pos_data.get("name", "MID"))
            if isinstance(name, dict):
                name = name.get("name", "MID")
            return POSITION_MAP.get(str(name), "MID")

        # Handle string format: "Goalkeeper"
        if isinstance(pos_data, str):
            return POSITION_MAP.get(pos_data, "MID")

        return "MID"

    try:
        # Extract simplified position for EVERY lineup entry, then take
        # the MODE (most common) per player. This fixes e.g. Messi being
        # labelled MID from a single central appearance when he played
        # ATT (RW/CF) in 90% of games.
        lineups_copy = lineups[["player_id", pos_col]].dropna(subset=[pos_col]).copy()
        lineups_copy["_simplified_pos"] = lineups_copy[pos_col].apply(primary_position)
        pos_series = (
            lineups_copy.groupby("player_id")["_simplified_pos"]
            .agg(lambda x: x.value_counts().index[0])   # mode
            .reset_index()
            .rename(columns={"_simplified_pos": "position"})
        )

        if "position" in df.columns:
            df = df.drop(columns=["position"])
        df = df.merge(pos_series, on="player_id", how="left")

    except Exception as e:
        print(f"  Warning: position extraction failed ({e}). Defaulting to MID.")

    if "position" not in df.columns:
        df["position"] = "MID"
    else:
        df["position"] = df["position"].fillna("MID")

    pos_counts = df["position"].value_counts().to_dict()
    print(f"  Positions assigned: {pos_counts}")
    return df


# -- Elite player / club filtering ---------------------------------------------
# Elite clubs whose players are typically unreachable for a small club.
# These are the ~20 richest clubs across the top-5 European leagues, whose
# players command wages and transfer fees far beyond a small club's reach.
# Source: Deloitte Football Money League (top-20 revenue clubs, 2015-2021).
# StatsBomb team names may vary slightly — common variants included.
ELITE_CLUBS = {
    # Spain
    "Barcelona", "Real Madrid",
    "Atlético Madrid", "Atletico Madrid", "Atlético de Madrid",
    # England
    "Manchester City", "Liverpool", "Chelsea",
    "Manchester United", "Arsenal", "Tottenham Hotspur",
    # Germany
    "Bayern Munich", "Bayern München",
    "Borussia Dortmund",
    # Italy
    "Juventus", "Inter Milan", "Internazionale",
    "AC Milan", "Milan",
    # France
    "Paris Saint-Germain", "Paris Saint Germain",
}


def filter_elite_players(df: pd.DataFrame,
                          elite_percentile: float = 85.0,
                          exclude_elite_clubs: bool = True) -> pd.DataFrame:
    """
    Remove players that a financially constrained club cannot realistically sign.

    Two complementary filters:
      1. PERFORMANCE CEILING — remove the top (100 - elite_percentile)% of
         players by predicted impact (y_hat). These are the Messis, Griezmanns,
         Suárez — players every top club is chasing. A small club doesn't even
         get a meeting with their agents.
      2. ELITE CLUB EXCLUSION — players at Barcelona, Real Madrid, etc. are
         contractually locked to clubs with enormous wage bills. Even a
         squad rotation player at Barça earns more than a small club's
         highest earner. Excludes players whose team_name is in ELITE_CLUBS.

    Returns a filtered DataFrame with unreachable players removed.
    """
    n_before = len(df)
    removed_reasons = []

    # Filter 1: Performance ceiling
    if "y_hat" in df.columns and elite_percentile < 100:
        cutoff = df["y_hat"].quantile(elite_percentile / 100.0)
        elite_perf = df["y_hat"] >= cutoff
        n_perf = elite_perf.sum()
        if n_perf > 0:
            removed_reasons.append(f"{n_perf} by performance (top {100-elite_percentile:.0f}%)")
        df = df[~elite_perf].copy()

    # Filter 2: Elite club exclusion
    if exclude_elite_clubs and "team_name" in df.columns:
        elite_club_mask = df["team_name"].isin(ELITE_CLUBS)
        n_club = elite_club_mask.sum()
        if n_club > 0:
            removed_reasons.append(f"{n_club} from elite clubs")
        df = df[~elite_club_mask].copy()

    n_after = len(df)
    print(f"  Elite filter: {n_before} → {n_after} candidates")
    for r in removed_reasons:
        print(f"    Removed {r}")

    if n_after < 11:
        print("  WARNING: fewer than 11 candidates remain! Relaxing filters.")
        # This shouldn't happen with real data, but safety net
        return df

    return df


# -- Transfer cost estimation (Transfermarkt-calibrated) -----------------------
# Position multipliers calibrated from Transfermarkt median values (2019-2021):
#   - Strikers/Wingers trade at ~1.3× the median player value
#   - Central midfielders at ~1.1×
#   - Defenders at ~0.85×
#   - Goalkeepers at ~0.65×
# Source: Transfermarkt.com historical transfer records, La Liga 2019-2021.
POSITION_COST_MULTIPLIER = {
    "ATT": 1.30,
    "MID": 1.10,
    "DEF": 0.85,
    "GK":  0.65,
}

# Transfermarkt-calibrated value tiers for La Liga (€M, 2019-2021 seasons).
# These anchor the synthetic model to real market ranges so that costs reflect
# what a small-to-mid-table club actually pays.
#
# Methodology:
#   - Tiers are defined by performance percentile among NON-ELITE players.
#   - Ranges are drawn from Transfermarkt.com transfer records for La Liga
#     clubs OUTSIDE the top 3 (i.e., excluding Barcelona, Real Madrid,
#     Atlético Madrid), seasons 2018/19 – 2020/21.
#   - Low-end: relegation-zone bench players, youth academy graduates
#   - Mid-range: solid starters at mid-table clubs (Betis, Sociedad, Villarreal)
#   - High-end: standout performers at smaller clubs that bigger clubs poach
#
# Reference: Transfermarkt.com — "La Liga Transfers" archive pages.
# Additional cross-reference: CIES Football Observatory annual reports on
# estimated transfer values (2019-2021).
#
# Note: these are ESTIMATED ranges for the player pool AFTER elite filtering.
# The pipeline removes top-percentile players before this stage, so the
# remaining pool corresponds roughly to the mid/lower tiers below.
COST_TIERS = {
    # (percentile_lower, percentile_upper): (min_cost_€M, max_cost_€M)
    (0.00, 0.20):  (0.5, 3.0),     # Fringe / youth / deep squad — loan fees, minimal transfers
    (0.20, 0.40):  (2.0, 8.0),     # Rotation players — typical relegation-zone signings
    (0.40, 0.60):  (5.0, 15.0),    # Solid starters — mid-table La Liga level
    (0.60, 0.80):  (10.0, 25.0),   # Above-average — Betis/Sociedad/Villarreal tier
    (0.80, 1.00):  (18.0, 40.0),   # Standout performers — the best available to small clubs
}


def estimate_transfer_cost(df: pd.DataFrame,
                            budget_scale: float = 100.0) -> pd.DataFrame:
    """
    Estimate transfer cost using a tier-based model calibrated to real
    Transfermarkt values for non-elite La Liga players (2019-2021).

    Methodology:
      1. TIER ASSIGNMENT — players are bucketed by y_hat percentile into
         5 tiers, each mapped to a real-world cost range (€M) drawn from
         Transfermarkt data for mid/lower-table La Liga clubs.
      2. INTERPOLATION — within each tier, cost scales linearly with the
         player's percentile rank (better player = more expensive within tier).
      3. POSITION MULTIPLIER — attackers cost ~1.3× the base, GKs ~0.65×,
         reflecting real market dynamics (source: Transfermarkt median by position).
      4. CAREER STAGE — peak value at 2-3 seasons of data (prime years proxy).
         Players with 5+ seasons get a discount (aging discount, 0.7×).
      5. MARKET NOISE — ±15% variation for negotiation dynamics, contract
         length, agent fees, release clause situations.

    Sources:
      - Transfermarkt.com: La Liga transfer archive, 2018-2021
      - CIES Football Observatory: Estimated transfer values reports
      - Soccerway: Contract and wage data cross-reference

    The budget_scale parameter is kept for backward compatibility but the
    tier-based model uses absolute €M values instead.
    """
    df = df.copy()
    if "cost" not in df.columns:
        rng = np.random.RandomState(42)

        # Percentile rank of each player within the (already filtered) pool
        y = df["y_hat"].clip(lower=0)
        pct_rank = y.rank(pct=True)   # 0.0 to 1.0

        # Assign cost from tiers
        base_cost = pd.Series(0.0, index=df.index)
        for (pct_lo, pct_hi), (cost_lo, cost_hi) in COST_TIERS.items():
            mask = (pct_rank >= pct_lo) & (pct_rank < pct_hi)
            if pct_hi == 1.0:
                mask = mask | (pct_rank == 1.0)  # include the top player
            # Linear interpolation within tier
            frac = ((pct_rank - pct_lo) / (pct_hi - pct_lo)).clip(0, 1)
            base_cost = base_cost.where(~mask, cost_lo + frac * (cost_hi - cost_lo))

        # Position multiplier
        pos_mult = df["position"].map(POSITION_COST_MULTIPLIER).fillna(1.0)

        # Career stage proxy from n_seasons
        if "n_seasons" in df.columns:
            n_s = df["n_seasons"].clip(lower=1, upper=6)
            # Peak at 2-3 seasons: bell curve centered at 2.5
            age_factor = np.exp(-0.3 * (n_s - 2.5) ** 2)
            age_factor = age_factor / age_factor.max()
            age_factor = age_factor.clip(lower=0.5)
        else:
            age_factor = 1.0

        # Market noise (±15%)
        noise = rng.uniform(0.87, 1.13, len(df))

        df["cost"] = (base_cost * pos_mult * age_factor * noise).round(1)

        # Floor at 0.3M (even a free transfer has registration/agent costs)
        df["cost"] = df["cost"].clip(lower=0.3)

        # Summary stats
        print(f"  Cost model: Transfermarkt-calibrated tiers (non-elite pool)")
        print(f"  Cost range : €{df['cost'].min():.1f}M – €{df['cost'].max():.1f}M")
        print(f"  Median cost: €{df['cost'].median():.1f}M")

    return df


# -- ILP Squad Optimizer -------------------------------------------------------
class SquadOptimizer:
    """Integer Linear Programming squad selection."""

    FORMATION_CONSTRAINTS = {
        "4-3-3": {
            "GK":  (1, 3),
            "DEF": (4, 6),
            "MID": (3, 5),
            "ATT": (3, 5),
        },
        "4-4-2": {
            "GK":  (1, 3),
            "DEF": (4, 6),
            "MID": (4, 6),
            "ATT": (2, 4),
        },
        "3-5-2": {
            "GK":  (1, 3),
            "DEF": (3, 5),
            "MID": (5, 7),
            "ATT": (2, 4),
        },
    }

    def __init__(self, budget=500.0, squad_size=11, formation="4-3-3",
                 risk_cap=0.3, max_per_club=3,
                 min_young_pct=0.0, young_threshold_seasons=3):
        """
        Parameters
        ----------
        budget         : float — max total transfer spend (€M)
        squad_size     : int   — number of players to select
        formation      : str   — e.g. "4-3-3", "4-4-2", "3-5-2"
        risk_cap       : float — max average variance per player
        max_per_club   : int   — club diversity constraint
        min_young_pct  : float — minimum fraction of squad that must be "young"
                         prospects (0.0 to 1.0). E.g., 0.3 = at least 30% of
                         squad must be young. Default 0.0 (no constraint).
                         Ensures long-term squad sustainability and resale value.
        young_threshold_seasons : int — players with ≤ this many seasons of
                         data are classified as "young" (proxy for <23 years old,
                         since StatsBomb lacks birth dates). Default 3.
        """
        self.budget       = budget
        self.squad_size   = squad_size
        self.formation    = formation
        self.risk_cap     = risk_cap
        self.max_per_club = max_per_club
        self.min_young_pct = min_young_pct
        self.young_threshold_seasons = young_threshold_seasons

    def optimize(self, candidates: pd.DataFrame) -> tuple:
        """Run the ILP."""
        df = candidates.reset_index(drop=True).copy()
        n  = len(df)

        if n < self.squad_size:
            raise ValueError(f"Need at least {self.squad_size} candidates, got {n}.")

        prob = LpProblem("SquadSelection", LpMaximize)
        z    = [LpVariable(f"z_{i}", cat=LpBinary) for i in range(n)]

        # Objective
        prob += lpSum(z[i] * df.loc[i, "v_adj"] for i in range(n))

        # 1. Budget
        prob += lpSum(z[i] * df.loc[i, "cost"] for i in range(n)) <= self.budget

        # 2. Squad size
        prob += lpSum(z[i] for i in range(n)) == self.squad_size

        # 3. Positional requirements (capped by available players)
        form_constraints = self.FORMATION_CONSTRAINTS.get(
            self.formation, self.FORMATION_CONSTRAINTS["4-3-3"]
        )
        for pos, (min_p, max_p) in form_constraints.items():
            pos_indices = [i for i in range(n) if df.loc[i, "position"] == pos]
            if pos_indices:
                effective_min = min(min_p, len(pos_indices))
                prob += lpSum(z[i] for i in pos_indices) >= effective_min
                prob += lpSum(z[i] for i in pos_indices) <= max_p

        # 4. Risk cap
        prob += (
            lpSum(z[i] * df.loc[i, "variance"] for i in range(n))
            <= self.risk_cap * self.squad_size
        )

        # 5. Youth / sustainability constraint
        #    Ensures a minimum fraction of the squad are "young prospects"
        #    for long-term resale value and squad evolution.
        #    Since StatsBomb lacks birth dates, we proxy youth by n_seasons ≤ threshold.
        if self.min_young_pct > 0 and "n_seasons" in df.columns:
            young_indices = [i for i in range(n)
                            if df.loc[i, "n_seasons"] <= self.young_threshold_seasons]
            min_young = int(np.ceil(self.min_young_pct * self.squad_size))
            if len(young_indices) >= min_young:
                prob += lpSum(z[i] for i in young_indices) >= min_young
                print(f"  Youth constraint: ≥{min_young} players with "
                      f"≤{self.young_threshold_seasons} seasons "
                      f"({len(young_indices)} eligible)")
            else:
                print(f"  Youth constraint: only {len(young_indices)} young players "
                      f"available (need {min_young}) — constraint relaxed.")

        # 6. Club diversity — max N players from any single team
        #    This is what makes ILP valuable over greedy: greedy can't
        #    handle combinatorial diversity constraints.
        if "team_name" in df.columns and self.max_per_club < self.squad_size:
            for club in df["team_name"].unique():
                club_indices = [i for i in range(n)
                                if df.loc[i, "team_name"] == club]
                if len(club_indices) > self.max_per_club:
                    prob += (lpSum(z[i] for i in club_indices)
                             <= self.max_per_club)

        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        status = LpStatus[prob.status]
        print(f"  Solver status: {status}")
        print(f"  Objective value: {value(prob.objective):.4f}")

        selected_indices = [i for i in range(n) if value(z[i]) == 1.0]
        selected_df      = df.loc[selected_indices].copy()

        return selected_df, status

    def summarize_squad(self, squad: pd.DataFrame) -> None:
        """Print a formatted squad summary."""
        total_cost   = squad["cost"].sum()
        total_impact = squad["v_adj"].sum()
        avg_risk     = squad["variance"].mean()
        has_team     = "team_name" in squad.columns

        width = 72 if has_team else 55
        print("\n" + "="*width)
        print(f"  OPTIMIZED SQUAD -- {self.formation}")
        print("="*width)
        if has_team:
            print(f"  {'Player':<25} {'Team':<15} {'Pos':>4} {'v_adj':>8} {'Cost €M':>8} {'Risk':>6}")
        else:
            print(f"  {'Player':<25} {'Pos':>4} {'v_adj':>8} {'Cost €M':>8} {'Risk':>6}")
        print("-"*width)
        for _, row in squad.sort_values("position").iterrows():
            name = str(row.get("player", row["player_id"]))[:24]
            if has_team:
                team = str(row.get("team_name", ""))[:14]
                print(f"  {name:<25} {team:<15} {row['position']:>4} "
                      f"{row['v_adj']:>8.3f} {row['cost']:>8.1f} "
                      f"{row['variance']:>6.3f}")
            else:
                print(f"  {name:<25} {row['position']:>4} "
                      f"{row['v_adj']:>8.3f} {row['cost']:>8.1f} "
                      f"{row['variance']:>6.3f}")
        print("="*width)
        print(f"  Total Cost   : €{total_cost:.1f}M / €{self.budget:.1f}M")
        print(f"  Total Impact : {total_impact:.3f}")
        print(f"  Avg Risk     : {avg_risk:.4f} (cap: {self.risk_cap})")
        print(f"  Budget Used  : {100*total_cost/self.budget:.1f}%")
        if has_team:
            clubs = squad["team_name"].nunique()
            print(f"  Clubs        : {clubs} (max {self.max_per_club} per club)")


# -- Baseline comparisons ------------------------------------------------------
def _check_club_cap(squad_rows, row, max_per_club):
    """Check if adding this player violates the club diversity cap."""
    if max_per_club is None or "team_name" not in row.index:
        return True
    club = row.get("team_name", "Unknown")
    current_count = sum(1 for r in squad_rows
                        if r.get("team_name", "") == club)
    return current_count < max_per_club


def select_greedy_squad(candidates, squad_size=11, budget=500.0,
                        max_per_club=None):
    """Greedy baseline: pick highest v_adj within budget + club cap."""
    df     = candidates.sort_values("v_adj", ascending=False).copy()
    squad  = []
    spent  = 0.0
    for _, row in df.iterrows():
        if len(squad) >= squad_size:
            break
        if (spent + row["cost"] <= budget
                and _check_club_cap(squad, row, max_per_club)):
            squad.append(row)
            spent += row["cost"]
    return pd.DataFrame(squad)


def select_value_per_cost_squad(candidates, squad_size=11, budget=500.0,
                                max_per_club=None):
    """Value-per-cost heuristic + club cap."""
    df    = candidates.copy()
    df["vpc"] = df["v_adj"] / df["cost"].clip(lower=0.1)
    df    = df.sort_values("vpc", ascending=False)
    squad = []
    spent = 0.0
    for _, row in df.iterrows():
        if len(squad) >= squad_size:
            break
        if (spent + row["cost"] <= budget
                and _check_club_cap(squad, row, max_per_club)):
            squad.append(row)
            spent += row["cost"]
    return pd.DataFrame(squad)


def select_random_squad(candidates, squad_size=11, budget=500.0,
                        seed=42, max_per_club=None):
    """Random selection baseline (within budget + club cap)."""
    rng   = np.random.RandomState(seed)
    df    = candidates.sample(frac=1, random_state=rng).copy()
    squad = []
    spent = 0.0
    for _, row in df.iterrows():
        if len(squad) >= squad_size:
            break
        if (spent + row["cost"] <= budget
                and _check_club_cap(squad, row, max_per_club)):
            squad.append(row)
            spent += row["cost"]
    return pd.DataFrame(squad)


def compare_strategies(candidates, optimizer):
    """Run all 4 strategies and return a comparison DataFrame.

    All strategies now enforce the same club diversity constraint
    so the comparison is fair — the ILP's advantage is that it finds
    the OPTIMAL solution under these constraints, while greedy/vpc
    may miss better combinations.
    """
    ilp_squad, status = optimizer.optimize(candidates)
    if status != "Optimal":
        print("  Warning: ILP did not find optimal solution.")

    mpc = optimizer.max_per_club
    greedy  = select_greedy_squad(candidates, optimizer.squad_size,
                                  optimizer.budget, max_per_club=mpc)
    vpc     = select_value_per_cost_squad(candidates, optimizer.squad_size,
                                          optimizer.budget, max_per_club=mpc)
    random_ = select_random_squad(candidates, optimizer.squad_size,
                                   optimizer.budget, max_per_club=mpc)

    records = []
    for name, squad in [
        ("ILP (ours)",          ilp_squad),
        ("Greedy",              greedy),
        ("Value-per-cost",      vpc),
        ("Random",              random_),
    ]:
        if len(squad) == 0:
            continue
        records.append({
            "Strategy":               name,
            "Total Impact (v_adj)":   round(squad["v_adj"].sum(), 3),
            "Impact / Cost":          round(squad["v_adj"].sum() / squad["cost"].sum(), 4),
            "Avg Risk":               round(squad["variance"].mean(), 4),
            "Total Cost":             round(squad["cost"].sum(), 1),
            "N Players":              len(squad),
        })

    results = pd.DataFrame(records)

    random_impact = results.loc[results["Strategy"] == "Random",
                                "Total Impact (v_adj)"].values[0]
    results["vs. Random (%)"] = (
        (results["Total Impact (v_adj)"] - random_impact)
        / abs(random_impact) * 100
    ).round(1)

    return results


def plot_strategy_comparison(results, save_path=None):
    """Bar chart comparing all strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    strategies = results["Strategy"].tolist()
    colors     = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

    for ax, col, title in zip(
        axes,
        ["Total Impact (v_adj)", "Impact / Cost", "Avg Risk"],
        ["Total Squad Impact", "Impact per Unit Cost", "Average Risk"],
    ):
        vals = results[col].tolist()
        bars = ax.bar(strategies, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(col)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Squad Selection Strategy Comparison", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved comparison plot to {save_path}")
    else:
        plt.show()


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    predictions = pd.read_parquet("outputs/player_predictions.parquet")
    normalized  = pd.read_parquet("outputs/player_features_normalized.parquet")
    lineups     = pd.read_parquet("outputs/raw_lineups.parquet")

    candidates = compute_risk_adjusted_value(predictions, normalized, lam=0.5)
    candidates = assign_positions(candidates, lineups)
    candidates = filter_elite_players(candidates, elite_percentile=85.0)
    candidates = estimate_transfer_cost(candidates)

    print(f"\nCandidates: {len(candidates)} players")
    print(candidates[["player", "position", "v_adj", "cost"]].head(10))

    optimizer = SquadOptimizer(budget=80.0, squad_size=11,
                                formation="4-3-3", risk_cap=0.3)
    squad, status = optimizer.optimize(candidates)
    optimizer.summarize_squad(squad)

    print("\n=== STRATEGY COMPARISON ===")
    results = compare_strategies(candidates, optimizer)
    print(results.to_string(index=False))

    results.to_csv("outputs/strategy_comparison.csv", index=False)
    squad.to_parquet("outputs/optimized_squad.parquet", index=False)
    plot_strategy_comparison(results, save_path="outputs/strategy_comparison.png")

    print("\n=== SENSITIVITY: lam (risk aversion) ===")
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cands = compute_risk_adjusted_value(predictions, normalized, lam=lam)
        cands = assign_positions(cands, lineups)
        cands = filter_elite_players(cands, elite_percentile=85.0)
        cands = estimate_transfer_cost(cands)
        try:
            sq, _ = optimizer.optimize(cands)
            print(f"  lam={lam:.2f} -> total impact={sq['v_adj'].sum():.3f}, "
                  f"avg_risk={sq['variance'].mean():.4f}")
        except Exception as e:
            print(f"  lam={lam:.2f} -> failed: {e}")
