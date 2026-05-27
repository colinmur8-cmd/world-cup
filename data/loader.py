import pandas as pd
from data.wc2018 import MATCHES_2018
from data.wc2022 import MATCHES_2022

_COLS = ["stage", "date", "home_team", "away_team", "home_goals", "away_goals"]


def load_matches(tournaments=("2018", "2022")) -> pd.DataFrame:
    rows = []
    if "2018" in tournaments:
        for r in MATCHES_2018:
            rows.append(("2018",) + r)
    if "2022" in tournaments:
        for r in MATCHES_2022:
            rows.append(("2022",) + r)

    df = pd.DataFrame(rows, columns=["tournament"] + _COLS)
    df["date"] = pd.to_datetime(df["date"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    return df.sort_values("date").reset_index(drop=True)


def all_teams(df: pd.DataFrame) -> list[str]:
    return sorted(set(df["home_team"]) | set(df["away_team"]))
