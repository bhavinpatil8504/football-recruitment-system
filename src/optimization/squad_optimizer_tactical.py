"""
Tactical Squad Optimizer — extension of the base ILP
-----------------------------------------------------
Same ILP structure but can optimize on different value columns
(v_adj, v_style) and includes tactical fit reporting.
"""

import numpy as np
import pandas as pd
from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                  LpBinary, LpStatus, value, PULP_CBC_CMD)


class TacticalSquadOptimizer:
    """
    ILP squad optimizer that supports tactical-fit-adjusted objectives.

    Same constraints as SquadOptimizer (budget, formation, risk cap),
    but the objective can be switched between:
      - v_adj   (pure impact, original)
      - v_style (blended impact + tactical fit)
    """

    FORMATION_CONSTRAINTS = {
        "4-3-3": {"GK": (1, 3), "DEF": (4, 6), "MID": (3, 5), "ATT": (3, 5)},
        "4-4-2": {"GK": (1, 3), "DEF": (4, 6), "MID": (4, 6), "ATT": (2, 4)},
        "3-5-2": {"GK": (1, 3), "DEF": (3, 5), "MID": (5, 7), "ATT": (2, 4)},
    }

    def __init__(self, budget=500.0, squad_size=11, formation="4-3-3",
                 risk_cap=0.3, max_per_club=3):
        self.budget       = budget
        self.squad_size   = squad_size
        self.formation    = formation
        self.risk_cap     = risk_cap
        self.max_per_club = max_per_club

    def optimize(self, candidates: pd.DataFrame,
                 value_col: str = "v_style") -> tuple:
        """
        Run the ILP with a configurable objective column.

        Parameters
        ----------
        candidates : DataFrame with at minimum: player_id, position,
                     cost, variance, and the value column
        value_col  : which column to maximize (default: v_style)
        """
        df = candidates.reset_index(drop=True).copy()
        n  = len(df)

        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found. Available: {list(df.columns)}")

        prob = LpProblem("TacticalSquadSelection", LpMaximize)
        z    = [LpVariable(f"z_{i}", cat=LpBinary) for i in range(n)]

        # Objective: maximize chosen value column
        prob += lpSum(z[i] * df.loc[i, value_col] for i in range(n))

        # Budget constraint
        prob += lpSum(z[i] * df.loc[i, "cost"] for i in range(n)) <= self.budget

        # Squad size
        prob += lpSum(z[i] for i in range(n)) == self.squad_size

        # Formation constraints
        form = self.FORMATION_CONSTRAINTS.get(
            self.formation, self.FORMATION_CONSTRAINTS["4-3-3"]
        )
        for pos, (min_p, max_p) in form.items():
            pos_idx = [i for i in range(n) if df.loc[i, "position"] == pos]
            if pos_idx:
                effective_min = min(min_p, len(pos_idx))
                prob += lpSum(z[i] for i in pos_idx) >= effective_min
                prob += lpSum(z[i] for i in pos_idx) <= max_p

        # Risk cap
        prob += (
            lpSum(z[i] * df.loc[i, "variance"] for i in range(n))
            <= self.risk_cap * self.squad_size
        )

        # Club diversity — max N players from any single team
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
        selected = [i for i in range(n) if value(z[i]) == 1.0]
        squad    = df.loc[selected].copy()

        return squad, status

    def summarize_tactical_squad(self, squad: pd.DataFrame,
                                  style_name: str = "Custom") -> None:
        """Print squad with tactical fit info."""
        total_cost   = squad["cost"].sum()
        total_impact = squad["v_adj"].sum() if "v_adj" in squad.columns else 0
        total_style  = squad["v_style"].sum() if "v_style" in squad.columns else 0
        mean_fit     = squad["tactical_fit"].mean() if "tactical_fit" in squad.columns else 0
        avg_risk     = squad["variance"].mean()

        print(f"\n{'='*65}")
        print(f"  TACTICAL SQUAD — {self.formation} — Style: {style_name}")
        print(f"{'='*65}")
        print(f"  {'Player':<25} {'Pos':>4} {'v_adj':>7} {'Fit':>5} "
              f"{'v_style':>8} {'Cost €M':>8}")
        print(f"{'-'*65}")
        for _, row in squad.sort_values("position").iterrows():
            name = str(row.get("player", row["player_id"]))[:24]
            fit  = row.get("tactical_fit", 0)
            vs   = row.get("v_style", row.get("v_adj", 0))
            print(f"  {name:<25} {row['position']:>4} {row['v_adj']:>7.3f} "
                  f"{fit:>5.2f} {vs:>8.3f} {row['cost']:>7.1f}")
        print(f"{'='*65}")
        print(f"  Total Cost     : €{total_cost:.1f}M / €{self.budget:.1f}M")
        print(f"  Total Impact   : {total_impact:.3f}")
        print(f"  Total v_style  : {total_style:.3f}")
        print(f"  Mean Tactic Fit: {mean_fit:.3f}")
        print(f"  Avg Risk       : {avg_risk:.4f}")
        print(f"  Budget Used    : {100*total_cost/self.budget:.1f}%")
        if "team_name" in squad.columns:
            clubs = squad["team_name"].nunique()
            print(f"  Clubs          : {clubs} (max {self.max_per_club} per club)")
