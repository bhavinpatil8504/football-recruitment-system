"""
Stage 4: Bayesian Cross-League Normalization  (pure numpy — no compiler needed)
--------------------------------------------------------------------------------
Hierarchical model for cross-league player comparison:

    y_{p,l,s}  ~  N(μ_{p,s} + δ_l,  σ_l²)

Where:
    y_{p,l,s} = observed metric for player p in league l, season s
    μ_{p,s}   = latent player skill (allowed to vary by season)
    δ_l       = league difficulty offset (harder leagues → positive δ)
    σ_l²      = within-league noise variance

Implementation: Empirical Bayes / James-Stein shrinkage (closed-form).
  Estimates σ_skill and σ_l from data (MLE), then computes the posterior
  mean of μ_{p,s} analytically. For N > 50 players, EB and full MCMC give
  nearly identical point estimates (Efron & Morris 1975).

  The transform() method computes a per-observation posterior (not a single
  per-player constant) so that a player's normalized score reflects actual
  season-to-season performance changes while still being league-adjusted.

Math:
  δ_l        = league_mean - grand_mean              (MLE, blended with prior)
  σ_l²       = within-league variance                (MLE)
  y_adj      = y_{p,l,s} - δ_l                      (league-adjusted observation)
  B          = σ_l² / (σ_l² + σ_skill²)             (shrinkage factor ∈ (0,1))
  μ_{p,s}    = (1 - B)·y_adj + B·μ_global           (posterior mean)
  var(μ_{p,s}) = (1 - B)·σ_l²                       (posterior variance)

  High B → heavy shrinkage (noisy league or few observations).
  Low  B → minimal shrinkage (stable league with many observations).

Note on league difficulty priors:
  LEAGUE_DIFFICULTY_PRIORS below are initial estimates blended with MLE.
  With sufficient data (n > min_obs_per_league), the MLE dominates (α → 1)
  and priors have negligible effect. The priors exist to regularize leagues
  with very few StatsBomb seasons (e.g., Ligue 1 has only 3 free seasons).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")


# ── League difficulty priors ──────────────────────────────────────────────────
# These priors are used ONLY for regularization when a league has very few
# seasons in the StatsBomb free dataset. With sufficient data (n > min_obs),
# the MLE estimate dominates via the blending weight α = n/(n + min_obs).
#
# Values are calibrated from UEFA 5-year coefficient rankings (2019–2024):
#   La Liga & PL at top → positive offsets (harder competition)
#   Eredivisie / Liga NOS → negative (lower competition intensity)
# The exact values matter little: with La Liga's ~18 seasons, α ≈ 0.98,
# so the prior contributes ~2% to La Liga's offset estimate.
LEAGUE_DIFFICULTY_PRIORS = {
    "La Liga":        0.00,   # reference level (most data in StatsBomb)
    "Premier League": 0.05,   # UEFA coeff: comparable to La Liga
    "Bundesliga":    -0.05,   # UEFA coeff: slightly below top 2
    "Serie A":       -0.03,   # UEFA coeff: close to Bundesliga
    "Ligue 1":       -0.10,   # UEFA coeff: 5th in big-5
    "Eredivisie":    -0.25,   # feeder league, large talent export
    "Championship":  -0.30,   # English 2nd tier
    "Liga NOS":      -0.20,   # Portuguese league (feeder)
    "unknown":        0.00,   # default: no adjustment
}

COMPETITION_TO_LEAGUE = {
    11: "La Liga",
    2:  "Premier League",
    9:  "Bundesliga",
    12: "Serie A",
    7:  "Ligue 1",
    13: "Eredivisie",
}


class BayesianLeagueNormalizer:
    """
    Empirical Bayes hierarchical normalizer.
    Drop-in replacement for the PyMC version — same API, same column names.
    Requires only numpy. No C compiler. Works on all platforms.

    Usage
    -----
    norm = BayesianLeagueNormalizer(target_metric="xg_sum_total")
    norm.fit(df, train_seasons=[1, 2, 3])
    df_out = norm.transform(df)
    # Adds columns: xg_sum_total_normalized, xg_sum_total_uncertainty
    """

    def __init__(self, target_metric: str = "xg_sum_total",
                 min_obs_per_league: int = 3,
                 winsorize_pct: float = 95.0):
        """
        Parameters
        ----------
        target_metric      : str   — column name of the metric to normalize
        min_obs_per_league : int   — minimum observations for full MLE trust
        winsorize_pct      : float — percentile above which to compress outliers
                             within each league. Addresses "dominant club effects"
                             where 2-3 top clubs in non-parity leagues inflate
                             individual stats. Default 95.0 (compress top 5%).
                             Set to 100.0 to disable.
        """
        self.target_metric      = target_metric
        self.min_obs_per_league = min_obs_per_league
        self.winsorize_pct      = winsorize_pct

        self.grand_mean_    = None
        self.sigma_skill_   = None
        self.league_stats_  = {}     # {league: {"delta": float, "sigma": float}}
        self.player_stats_  = {}     # {player_id: {"mu": float, "var": float}}
        self.y_mean_        = None
        self.y_std_         = None

    def _league_name(self, comp_id) -> str:
        try:
            return COMPETITION_TO_LEAGUE.get(int(comp_id), "unknown")
        except (ValueError, TypeError):
            return "unknown"

    def fit(self, df: pd.DataFrame,
            train_seasons: Optional[List] = None) -> "BayesianLeagueNormalizer":
        """
        Estimate all model parameters from training data only.

        Parameters
        ----------
        df            : full player-season DataFrame
        train_seasons : season_ids to fit on (leakage guard).
                        If None, uses all seasons.
        """
        data = df.copy()

        if train_seasons is not None:
            n_before = len(data)
            data = data[data["season_id"].isin(train_seasons)]
            print(f"  Leakage guard: fitting on seasons {sorted(train_seasons)} "
                  f"({len(data)} rows, excluded {n_before - len(data)} test rows)")

        data = data.dropna(subset=["player_id", "competition_id",
                                    self.target_metric])

        if len(data) < 5:
            print(f"  Warning: only {len(data)} rows — skipping normalization.")
            self.grand_mean_ = 0.0
            self.sigma_skill_ = 1.0
            self.y_mean_ = 0.0
            self.y_std_  = 1.0
            return self

        data["league"] = data["competition_id"].apply(self._league_name)
        y_raw = data[self.target_metric].values.astype(float)

        # Step 1: z-score standardisation so priors operate on unit scale
        self.y_mean_ = float(y_raw.mean())
        self.y_std_  = float(y_raw.std()) + 1e-9
        data = data.copy()
        data["_y"] = (y_raw - self.y_mean_) / self.y_std_

        # Step 1b: Within-league quantile compression (dominant-club bias fix)
        # In non-parity leagues (e.g., Portuguese Liga, Eredivisie, or even
        # La Liga where Barça/Real dominate), 2-3 clubs inflate individual
        # stats. Winsorize per league at the specified percentile to prevent
        # dominant-club outliers from skewing league-level estimates.
        if self.winsorize_pct < 100.0:
            n_clipped = 0
            for league, grp_idx in data.groupby("league").groups.items():
                y_league = data.loc[grp_idx, "_y"]
                cap = float(np.percentile(y_league, self.winsorize_pct))
                mask = y_league > cap
                if mask.any():
                    n_clipped += mask.sum()
                    data.loc[grp_idx[mask.values], "_y"] = cap
            if n_clipped > 0:
                print(f"  Quantile compression: clipped {n_clipped} outliers "
                      f"at p{self.winsorize_pct:.0f} per league")

        self.grand_mean_ = float(data["_y"].mean())

        # Step 2: League difficulty offsets (MLE, blended with prior)
        for league, grp in data.groupby("league"):
            n         = len(grp)
            delta_mle = float(grp["_y"].mean()) - self.grand_mean_
            sigma_mle = float(grp["_y"].std()) + 1e-9
            # Blend weight: more data → trust MLE more
            alpha     = min(n / (n + self.min_obs_per_league), 1.0)
            prior_d   = LEAGUE_DIFFICULTY_PRIORS.get(league, 0.0)
            self.league_stats_[league] = {
                "delta": alpha * delta_mle + (1 - alpha) * prior_d,
                "sigma": sigma_mle,
                "n":     n,
            }

        # Step 3: Between-player variance
        total_var   = float(data["_y"].var())
        mean_within = float(np.mean([v["sigma"]**2
                                     for v in self.league_stats_.values()]))
        # Floor sigma_skill at 10% of total std to prevent over-shrinkage.
        # With single-league data, total_var ≈ mean_within, making sigma_skill
        # near-zero. This causes B → 1 and collapses all posteriors to the
        # grand mean. A 10% floor preserves meaningful player differences.
        sigma_skill_raw = float(np.sqrt(max(total_var - mean_within, 1e-6)))
        sigma_skill_floor = float(np.sqrt(total_var)) * 0.10
        self.sigma_skill_ = max(sigma_skill_raw, sigma_skill_floor)

        # Step 4: Per-player James-Stein posterior
        for player_id, pgrp in data.groupby("player_id"):
            estimates = []
            for _, row in pgrp.iterrows():
                league   = row["league"]
                stats    = self.league_stats_.get(
                    league, {"delta": 0.0, "sigma": 1.0}
                )
                sigma_l  = stats["sigma"]
                y_adj    = float(row["_y"]) - stats["delta"]
                B        = sigma_l**2 / (sigma_l**2 + self.sigma_skill_**2 + 1e-9)
                mu_p     = (1 - B) * y_adj + B * self.grand_mean_
                var_p    = (1 - B) * sigma_l**2
                estimates.append((mu_p, var_p))

            self.player_stats_[player_id] = {
                "mu":  float(np.mean([e[0] for e in estimates])),
                "var": float(np.mean([e[1] for e in estimates])),
            }

        print(f"  Fitted: {len(self.league_stats_)} leagues, "
              f"{len(self.player_stats_)} players  "
              f"(σ_skill={self.sigma_skill_:.3f}, "
              f"grand_mean={self.grand_mean_:.3f})")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add normalized skill and uncertainty columns to all players.

        IMPORTANT: computes a posterior for EACH ROW using that row's
        raw metric value, not a single per-player constant. This ensures
        a player's normalized score varies across seasons (reflecting
        actual performance changes), while still being league-adjusted
        and shrunk toward the population mean.

        Model parameters (grand_mean_, sigma_skill_, league_stats_) are
        learned in fit() on training data only — no leakage.

        Adds:
          {metric}_normalized   → posterior mean μ_p (per observation)
          {metric}_uncertainty  → posterior std (higher = riskier signing)
        """
        if self.grand_mean_ is None:
            raise RuntimeError("Call .fit() before .transform()")

        out = df.copy()
        out["_league"] = out["competition_id"].apply(self._league_name)

        norm_col = f"{self.target_metric}_normalized"
        unc_col  = f"{self.target_metric}_uncertainty"

        mu_vals, var_vals = [], []
        for _, row in out.iterrows():
            league = row.get("_league", "unknown")
            stats  = self.league_stats_.get(
                league, {"delta": 0.0, "sigma": 1.0}
            )
            sigma_l = stats["sigma"]

            # Get this row's raw metric value and standardize it
            raw_val = row.get(self.target_metric, self.y_mean_)
            if pd.isna(raw_val):
                raw_val = self.y_mean_
            y_std = (float(raw_val) - self.y_mean_) / self.y_std_

            # League-adjust: remove league difficulty offset
            y_adj = y_std - stats["delta"]

            # James-Stein shrinkage toward grand mean
            B     = sigma_l**2 / (sigma_l**2 + self.sigma_skill_**2 + 1e-9)
            mu_p  = (1 - B) * y_adj + B * self.grand_mean_
            var_p = (1 - B) * sigma_l**2

            mu_vals.append(float(mu_p))
            var_vals.append(float(var_p))

        out[norm_col] = mu_vals
        out[unc_col]  = np.sqrt(np.array(var_vals))
        return out.drop(columns=["_league"], errors="ignore")

    def plot_league_effects(self, save_path: str = None):
        """Bar chart of league difficulty offsets."""
        if not self.league_stats_:
            return
        leagues = list(self.league_stats_.keys())
        deltas  = [self.league_stats_[l]["delta"] for l in leagues]

        fig, ax = plt.subplots(figsize=(9, 4))
        colors  = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
        ax.barh(leagues, deltas, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"δ_l  (metric: {self.target_metric})")
        ax.set_title("Cross-League Difficulty Estimates (Empirical Bayes)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved: {save_path}")
        else:
            plt.show()
        plt.close()


# ── Convenience wrapper used by pipeline.py ───────────────────────────────────
def normalize_all_metrics(player_df: pd.DataFrame,
                           metrics: List[str],
                           train_seasons: Optional[List] = None,
                           ) -> pd.DataFrame:
    """
    Normalize each metric using Empirical Bayes (James-Stein shrinkage).

    For each metric, fits a BayesianLeagueNormalizer on training seasons
    only (leakage guard), then transforms all rows. Produces:
      - {metric}_normalized  : posterior mean (league-adjusted, shrunk)
      - {metric}_uncertainty : posterior std (used in risk adjustment)

    Saves a league-effects plot for each metric to outputs/.
    """
    import os
    os.makedirs("outputs", exist_ok=True)

    result = player_df.copy()
    for metric in metrics:
        if metric not in player_df.columns:
            print(f"  Skipping '{metric}' — column not found")
            continue
        print(f"\n  Normalizing: {metric}")
        norm = BayesianLeagueNormalizer(target_metric=metric)
        norm.fit(result, train_seasons=train_seasons)
        result = norm.transform(result)
        norm.plot_league_effects(
            save_path=f"outputs/league_effects_{metric}.png"
        )
    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    features = pd.read_parquet("outputs/player_features.parquet")
    seasons  = sorted(features["season_id"].unique())
    train_s  = seasons[:-1]

    metrics = [
        "xg_sum_total",
        "progressive_passes_total",
        "key_passes_total",
        "pressure_count_total",
    ]

    print(f"Seasons: {seasons}  |  Training on: {train_s}")
    normalized = normalize_all_metrics(features, metrics, train_seasons=train_s)
    normalized.to_parquet("outputs/player_features_normalized.parquet",
                          index=False)
    print(f"\nDone. Shape: {normalized.shape}")
    print([c for c in normalized.columns if "normalized" in c or "uncertainty" in c])
