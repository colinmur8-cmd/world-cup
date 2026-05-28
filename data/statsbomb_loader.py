"""
StatsBomb Open Data loader.

Extracts per-match team statistics (corners, shots, xG, cards, fouls)
from StatsBomb event-level data for all available free international competitions.

Free competitions used:
  - FIFA World Cup 2018 + 2022  (comp 43)
  - Copa America 2024            (comp 223)
  - African Cup of Nations 2023  (comp 1267)

Resulting cache: data/_statsbomb_stats_cache.csv
"""

from __future__ import annotations
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

# (competition_id, season_id, display_name, year_string)
INTERNATIONAL_COMPETITIONS = [
    (43,   3,   "FIFA World Cup",          "2018"),
    (43,   106, "FIFA World Cup",          "2022"),
    (223,  282, "Copa America",            "2024"),
    (1267, 107, "African Cup of Nations",  "2023"),
]

_DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "_statsbomb_stats_cache.csv")

# StatsBomb team name → our canonical name
SB_ALIASES: dict[str, str] = {
    "United States":   "USA",
    "Korea Republic":  "South Korea",
    "IR Iran":         "Iran",
    "Côte d'Ivoire":   "Ivory Coast",
    "Czech Republic":  "Czechia",
    "Wales":           "Wales",
    "Bosnia and Herzegovina": "Bosnia",
    "DR Congo":        "DR Congo",
    "Cape Verde":      "Cape Verde",
}


def _norm(name: str) -> str:
    return SB_ALIASES.get(name, name)


def _extract_match_stats(match_id: int, sb_home: str, sb_away: str) -> dict | None:
    """
    Load events for one match and return a flat dict of per-team stats.
    Returns None on failure.
    """
    from statsbombpy import sb

    try:
        events = sb.events(match_id=match_id, fmt="dataframe")
    except Exception:
        return None

    result = {}
    for side, sb_team in (("home", sb_home), ("away", sb_away)):
        te = events[events["team"] == sb_team]

        # Corners — pass events with pass_type == 'Corner'
        corners = te[te["pass_type"] == "Corner"]

        # Shots
        shots = te[te["type"] == "Shot"]
        on_target = shots[
            shots["shot_outcome"].isin(["Goal", "Saved", "Saved To Post", "Saved to Post"])
        ]

        # xG
        xg = float(shots["shot_statsbomb_xg"].fillna(0.0).sum())

        # Cards: from foul events
        yellow = te[te["foul_committed_card"] == "Yellow Card"]
        red    = te[te["foul_committed_card"] == "Red Card"]

        # Fouls committed
        fouls = te[te["type"] == "Foul Committed"]

        result[f"{side}_corners"]          = len(corners)
        result[f"{side}_shots"]            = len(shots)
        result[f"{side}_shots_on_target"]  = len(on_target)
        result[f"{side}_xg"]               = round(xg, 4)
        result[f"{side}_yellow"]           = len(yellow)
        result[f"{side}_red"]              = len(red)
        result[f"{side}_fouls"]            = len(fouls)

    return result


def load_statsbomb_match_stats(
    competitions: list | None = None,
    cache_path: str | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load (or rebuild) per-match statistics from StatsBomb open data.

    Returns a DataFrame with columns:
      competition, year, date, home_team, away_team,
      home_corners, away_corners, home_shots, away_shots,
      home_shots_on_target, away_shots_on_target,
      home_xg, away_xg, home_yellow, away_yellow,
      home_red, away_red, home_fouls, away_fouls
    """
    cache = cache_path or _DEFAULT_CACHE

    if os.path.exists(cache) and not refresh:
        return pd.read_csv(cache, parse_dates=["date"])

    from statsbombpy import sb

    comps = competitions or INTERNATIONAL_COMPETITIONS
    rows: list[dict] = []

    for comp_id, season_id, comp_name, year in comps:
        if verbose:
            print(f"  [{comp_name} {year}] fetching match list…", flush=True)
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
            continue

        n = len(matches)
        for idx, (_, match) in enumerate(matches.iterrows(), 1):
            if verbose:
                print(f"\r    {idx}/{n} matches processed…", end="", flush=True)

            sb_home = match["home_team"]
            sb_away = match["away_team"]
            stats   = _extract_match_stats(match["match_id"], sb_home, sb_away)

            if stats is None:
                continue

            rows.append({
                "competition": comp_name,
                "year":        year,
                "date":        match["match_date"],
                "home_team":   _norm(sb_home),
                "away_team":   _norm(sb_away),
                **stats,
            })

        if verbose:
            print(f"\r    {n}/{n} matches — done.              ")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(cache, index=False)
        if verbose:
            print(f"  Cached {len(df)} matches → {cache}")

    return df


# ── Quick summary ─────────────────────────────────────────────────────────────

def statsbomb_summary(df: pd.DataFrame) -> None:
    """Print a brief summary of the cached StatsBomb stats."""
    print(f"\n  StatsBomb stats cache: {len(df)} matches")
    for comp, grp in df.groupby("competition"):
        print(f"    {comp}: {len(grp)} matches "
              f"| corners/match: {(grp['home_corners']+grp['away_corners']).mean():.1f} "
              f"| shots/match: {(grp['home_shots']+grp['away_shots']).mean():.1f} "
              f"| yellows/match: {(grp['home_yellow']+grp['away_yellow']).mean():.1f}")


if __name__ == "__main__":
    print("Building StatsBomb stats cache…")
    df = load_statsbomb_match_stats(refresh=True)
    statsbomb_summary(df)
