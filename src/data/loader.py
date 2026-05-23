"""
Stage 1: Data Loading
---------------------
Loads match event data from StatsBomb Open Data.
StatsBomb provides free event-level data for several competitions
including La Liga (Messi years), Women's World Cup, EURO 2020, etc.

Run `pip install statsbombpy socceraction` before using this module.
"""

import pandas as pd
import numpy as np
from statsbombpy import sb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ── Available free competitions (updated from StatsBomb open-data repo) ──────
# Source: https://github.com/statsbomb/open-data/blob/master/data/competitions.json
FREE_COMPETITIONS = {
    # ── TOP 5 LEAGUES ──────────────────────────────────────────────────────
    "La Liga": {"competition_id": 11, "season_ids": [
        90, 42, 4, 1, 2, 27, 26, 25, 24, 23, 22, 21, 41, 40, 39, 38, 37, 278
        # 2020/21 … 2004/05, plus 1973/74 (278)
    ]},
    "Premier League": {"competition_id": 2, "season_ids": [
        27, 44        # 2015/16, 2003/04
    ]},
    "1. Bundesliga": {"competition_id": 9, "season_ids": [
        281, 27       # 2023/24, 2015/16
    ]},
    "Serie A": {"competition_id": 12, "season_ids": [
        27, 86        # 2015/16, 1986/87
    ]},
    "Ligue 1": {"competition_id": 7, "season_ids": [
        235, 108, 27  # 2022/23, 2021/22, 2015/16
    ]},

    # ── UEFA CLUB COMPETITIONS ─────────────────────────────────────────────
    "Champions League": {"competition_id": 16, "season_ids": [
        4, 42, 1, 2, 27, 26, 25, 24, 23, 22, 21, 41, 40, 39, 6, 44, 76, 78
        # 2018/19 backwards to 1970/71
    ]},

    # ── INTERNATIONAL TOURNAMENTS ──────────────────────────────────────────
    "FIFA World Cup": {"competition_id": 43, "season_ids": [
        106, 3, 30, 26, 23, 7, 53, 107
        # 2022, 2018, 1990, 1986, 1974, 1970, 1962, 1958
    ]},
    "UEFA Euro": {"competition_id": 55, "season_ids": [
        282, 43       # 2024, 2020
    ]},
    "Copa America": {"competition_id": 223, "season_ids": [253]},  # 2024

    # ── OTHER ──────────────────────────────────────────────────────────────
    "Indian Super League": {"competition_id": 1238, "season_ids": [108]},  # 2021/22
    "MLS": {"competition_id": 44, "season_ids": [282]},  # 2023
    "Copa del Rey": {"competition_id": 87, "season_ids": [40, 38, 37]},

    # ── WOMEN'S (separate pool) ────────────────────────────────────────────
    "FA Women's Super League": {"competition_id": 37, "season_ids": [4, 42, 90]},
    "NWSL": {"competition_id": 49, "season_ids": [3]},
    "Women's World Cup": {"competition_id": 72, "season_ids": [30, 107]},
}

# ── Curated pairs for top-5 league multi-league loading ──────────────────────
# These are the competition-season pairs that give the best coverage of the
# top 5 European leagues. Ordered by data richness and relevance.
TOP5_LEAGUE_PAIRS = [
    # Premier League
    (2, 27),     # PL 2015/16
    (2, 44),     # PL 2003/04
    # Bundesliga
    (9, 27),     # Bundesliga 2015/16
    (9, 281),    # Bundesliga 2023/24
    # Serie A
    (12, 27),    # Serie A 2015/16
    # Ligue 1
    (7, 27),     # Ligue 1 2015/16
    (7, 108),    # Ligue 1 2021/22
    (7, 235),    # Ligue 1 2022/23
]

# Champions League seasons — rich source of multi-national players
CHAMPIONS_LEAGUE_PAIRS = [
    (16, 4),     # CL 2018/19
    (16, 42),    # CL 2019/20
    (16, 1),     # CL 2017/18
    (16, 2),     # CL 2016/17
    (16, 27),    # CL 2015/16
    (16, 26),    # CL 2014/15
    (16, 25),    # CL 2013/14
    (16, 24),    # CL 2012/13
    (16, 23),    # CL 2011/12
    (16, 22),    # CL 2010/11
    (16, 21),    # CL 2009/10
    (16, 41),    # CL 2008/09
    (16, 40),    # CL 2007/08
    (16, 39),    # CL 2006/07
]

# International tournaments — supplements player diversity
INTERNATIONAL_PAIRS = [
    (55, 43),    # UEFA Euro 2020
    (55, 282),   # UEFA Euro 2024
    (43, 3),     # FIFA World Cup 2018
    (43, 106),   # FIFA World Cup 2022
]


def list_available_competitions() -> pd.DataFrame:
    """Print all free StatsBomb competitions."""
    comps = sb.competitions()
    print(f"Found {len(comps)} available competitions")
    print(comps[["competition_id", "season_id", "competition_name",
                  "season_name", "competition_gender"]].to_string())
    return comps


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Load all matches for a given competition + season."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"  Loaded {len(matches)} matches for "
          f"competition={competition_id}, season={season_id}")
    return matches


def load_events_for_season(competition_id: int,
                            season_id: int,
                            max_matches: int = None) -> pd.DataFrame:
    """
    Load all events for every match in a season.
    
    Parameters
    ----------
    competition_id : int
    season_id      : int
    max_matches    : int, optional
        Limit to first N matches (useful for quick testing).
    
    Returns
    -------
    pd.DataFrame with all events, augmented with match metadata.
    """
    matches = load_matches(competition_id, season_id)

    if max_matches:
        matches = matches.head(max_matches)
        print(f"  (Limited to {max_matches} matches for testing)")

    all_events = []
    for _, match in tqdm(matches.iterrows(),
                         total=len(matches),
                         desc="Loading match events"):
        try:
            events = sb.events(match_id=match["match_id"])
            # Attach useful match-level context
            events["match_id"]         = match["match_id"]
            events["match_date"]       = match["match_date"]
            events["competition_id"]   = competition_id
            events["season_id"]        = season_id
            events["home_team"]        = match["home_team"]
            events["away_team"]        = match["away_team"]
            events["home_score"]       = match["home_score"]
            events["away_score"]       = match["away_score"]
            all_events.append(events)
        except Exception as e:
            print(f"  Warning: failed to load match {match['match_id']}: {e}")

    if not all_events:
        raise ValueError("No events loaded. Check competition/season IDs.")

    df = pd.concat(all_events, ignore_index=True)
    print(f"\n  Total events loaded: {len(df):,}")
    return df


def load_lineups_for_season(competition_id: int,
                             season_id: int,
                             max_matches: int = None) -> pd.DataFrame:
    """Load lineup (player roster) data for every match."""
    matches = load_matches(competition_id, season_id)

    if max_matches:
        matches = matches.head(max_matches)

    all_lineups = []
    for _, match in tqdm(matches.iterrows(),
                         total=len(matches),
                         desc="Loading lineups"):
        try:
            lineup = sb.lineups(match_id=match["match_id"])
            for team_name, team_df in lineup.items():
                team_df["match_id"]       = match["match_id"]
                team_df["match_date"]     = match["match_date"]
                team_df["competition_id"] = competition_id
                team_df["season_id"]      = season_id
                team_df["team_name"]      = team_name
                all_lineups.append(team_df)
        except Exception as e:
            print(f"  Warning: failed lineup for match {match['match_id']}: {e}")

    return pd.concat(all_lineups, ignore_index=True)


def load_multi_season(competition_season_pairs: list,
                      max_matches_per_season: int = None) -> dict:
    """
    Load events and lineups for multiple seasons.

    Parameters
    ----------
    competition_season_pairs : list of (competition_id, season_id) tuples
    max_matches_per_season   : int, optional  — for quick iteration

    Returns
    -------
    dict with keys 'events' and 'lineups', each a pd.DataFrame
    """
    all_events  = []
    all_lineups = []

    for comp_id, season_id in competition_season_pairs:
        print(f"\nLoading competition={comp_id}, season={season_id}")
        try:
            events  = load_events_for_season(comp_id, season_id,
                                             max_matches=max_matches_per_season)
            lineups = load_lineups_for_season(comp_id, season_id,
                                              max_matches=max_matches_per_season)
            all_events.append(events)
            all_lineups.append(lineups)
        except Exception as e:
            print(f"  Skipping season due to error: {e}")

    return {
        "events":  pd.concat(all_events,  ignore_index=True),
        "lineups": pd.concat(all_lineups, ignore_index=True),
    }


def get_player_season_map(lineups: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean player → (team, competition, season) mapping
    from lineup data.
    """
    cols = ["player_id", "player_name", "team_name",
            "competition_id", "season_id"]
    player_map = (
        lineups[cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return player_map


# ── Quick-start helper ────────────────────────────────────────────────────────
def quick_load_laliga(n_seasons: int = 3,
                      max_matches_per_season: int = 20) -> dict:
    """
    Convenience function: load the first N La Liga seasons from StatsBomb.
    Good for initial development / testing on a laptop.

    La Liga season IDs (StatsBomb): 
        90 = 2020/21, 42 = 2019/20, 4 = 2018/19, 1 = 2017/18 ...
    """
    # StatsBomb free-tier La Liga seasons (newest first)
    LALIGA_SEASONS = [
        (11, 90),   # La Liga 2020/21
        (11, 42),   # La Liga 2019/20
        (11, 4),    # La Liga 2018/19
        (11, 1),    # La Liga 2017/18
        (11, 2),    # La Liga 2016/17
        (11, 27),   # La Liga 2015/16
        (11, 26),   # La Liga 2014/15
        (11, 25),   # La Liga 2013/14
        (11, 23),   # La Liga 2012/13
        (11, 22),   # La Liga 2011/12
        (11, 21),   # La Liga 2010/11
        (11, 41),   # La Liga 2009/10
        (11, 40),   # La Liga 2008/09
        (11, 38),   # La Liga 2007/08
        (11, 37),   # La Liga 2006/07
    ]
    pairs = LALIGA_SEASONS[:n_seasons]
    print(f"Quick-loading {n_seasons} La Liga seasons "
          f"({max_matches_per_season} matches each)...")
    return load_multi_season(pairs, max_matches_per_season=max_matches_per_season)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: load 2 seasons, 10 matches each (fast, for testing)
    data = quick_load_laliga(n_seasons=2, max_matches_per_season=10)

    events  = data["events"]
    lineups = data["lineups"]

    print(f"\nEvents shape  : {events.shape}")
    print(f"Lineups shape : {lineups.shape}")
    print(f"Event types   : {events['type'].value_counts().head(10).to_dict()}")

    player_map = get_player_season_map(lineups)
    print(f"Unique players: {player_map['player_id'].nunique()}")

    # Save for next stages
    events.to_parquet("outputs/raw_events.parquet", index=False)
    lineups.to_parquet("outputs/raw_lineups.parquet", index=False)
    player_map.to_parquet("outputs/player_map.parquet", index=False)
    print("\nSaved to outputs/")
