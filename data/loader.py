import pandas as pd
from data.wc2018 import MATCHES_2018
from data.wc2022 import MATCHES_2022

_COLS = ["stage", "date", "home_team", "away_team", "home_goals", "away_goals"]

# Standard team-name aliases: map external/alternate spellings to our canonical names.
TEAM_ALIASES: dict[str, str] = {
    "Korea Republic":     "South Korea",
    "Republic of Korea":  "South Korea",
    "IR Iran":            "Iran",
    "United States":      "USA",
    "Côte d'Ivoire":      "Ivory Coast",
    "Cote d'Ivoire":      "Ivory Coast",
    "Czech Republic":     "Czechia",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df["home_team"] = df["home_team"].replace(TEAM_ALIASES)
    df["away_team"] = df["away_team"].replace(TEAM_ALIASES)
    return df


def load_matches(tournaments=("2018", "2022")) -> pd.DataFrame:
    """Load hardcoded WC match data for the requested tournament years."""
    rows = []
    if "2018" in tournaments:
        for r in MATCHES_2018:
            rows.append(("2018",) + r)
    if "2022" in tournaments:
        for r in MATCHES_2022:
            rows.append(("2022",) + r)

    df = pd.DataFrame(rows, columns=["tournament"] + _COLS)
    df["date"]       = pd.to_datetime(df["date"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    return _normalise(df.sort_values("date").reset_index(drop=True))


def load_from_csv(csv_path: str,
                  tournaments: tuple[str, ...] = ("2018", "2022")) -> pd.DataFrame:
    """
    Load WC matches from an external CSV (e.g. Kaggle international results,
    football-data.org export, or fetch_wc.py output).

    Expected columns (flexible naming):
      date, home_team, away_team, home_score/home_goals, away_score/away_goals,
      tournament (optional — filtered by year if absent)
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    rename = {
        "home_score": "home_goals", "away_score": "away_goals",
        "score1":     "home_goals", "score2":     "away_goals",
        "team1":      "home_team",  "team2":      "away_team",
    }
    df.rename(columns=rename, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    years = [int(y) for y in tournaments]

    if "tournament" in df.columns:
        mask = (
            df["tournament"].str.upper().str.contains("WORLD CUP", na=False) &
            df["date"].dt.year.isin(years)
        )
        df = df[mask].copy()
    else:
        df = df[df["date"].dt.year.isin(years)].copy()

    required = {"date","home_team","away_team","home_goals","away_goals"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["home_goals"]  = df["home_goals"].astype(int)
    df["away_goals"]  = df["away_goals"].astype(int)
    df["total_goals"] = df["home_goals"] + df["away_goals"]

    # Assign tournament year from date if not present
    if "tournament" not in df.columns:
        df["tournament"] = df["date"].dt.year.astype(str)
    else:
        df["tournament"] = df["date"].dt.year.astype(str)

    # Stage column: assign "Group X" / "R16" / "QF" / "SF" / "Final" if missing
    if "stage" not in df.columns:
        df["stage"] = "Unknown"

    return _normalise(
        df[["tournament","stage","date","home_team","away_team",
            "home_goals","away_goals","total_goals"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def all_teams(df: pd.DataFrame) -> list[str]:
    return sorted(set(df["home_team"]) | set(df["away_team"]))
