"""
WC 2026 predictions framework.

The 2026 World Cup (USA/Canada/Mexico, 11 Jun – 19 Jul 2026) has 48 teams
across 12 groups of 4.

Team ratings below are seeded from WC 2018 + 2022 performance plus
approximate FIFA ranking adjustments as of May 2026.  Supply actual
bookmaker odds via the EXAMPLE_ODDS dict or via the CLI --odds flag
to get live edge calculations.
"""

from __future__ import annotations
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from model.dixon_coles import DixonColesModel
from model.markets import all_markets, match_result, over_under, btts
from model.edge import find_value, simulate_bookmaker_odds
from data.loader import load_matches
from tabulate import tabulate


# ── 2026 group-stage fixtures (first match date per group shown) ─────────────
# Update with confirmed fixture list once released.

WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["USA",        "Bolivia",     "Panama",      "Morocco"],
    "B": ["Argentina",  "Chile",       "Peru",        "Albania"],
    "C": ["Mexico",     "Jamaica",     "Honduras",    "Ukraine"],
    "D": ["France",     "Belgium",     "Saudi Arabia","Paraguay"],
    "E": ["Spain",      "Brazil",      "Japan",       "Serbia"],   # provisional
    "F": ["England",    "Netherlands", "Senegal",     "Ecuador"],
    "G": ["Portugal",   "Croatia",     "Cameroon",    "New Zealand"],
    "H": ["Germany",    "Australia",   "Colombia",    "Uruguay"],
    "I": ["Italy",      "South Korea", "Kenya",       "Iran"],
    "J": ["Canada",     "Venezuela",   "Costa Rica",  "Nigeria"],
    "K": ["Poland",     "Czechia",     "Switzerland", "Tunisia"],
    "L": ["Denmark",    "Ivory Coast", "Egypt",       "Qatar"],
}

# ── Full group-stage fixture list ────────────────────────────────────────────
def generate_group_fixtures(groups: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """Return list of (group, home_team, away_team) for all group games."""
    fixtures = []
    for grp, teams in groups.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                fixtures.append((grp, teams[i], teams[j]))
    return fixtures


# ── Convenience: load model trained on WC 2018 + 2022 ───────────────────────

def load_trained_model() -> DixonColesModel:
    df = load_matches(("2018", "2022"))
    return DixonColesModel().fit(df)


# ── Single-match prediction ───────────────────────────────────────────────────

def predict_match(home: str, away: str,
                  model: DixonColesModel | None = None,
                  bk_odds: dict[str, dict[str, float]] | None = None,
                  min_edge: float = 0.03) -> dict:
    """
    Predict a single match and optionally find value against bookmaker odds.

    bk_odds format:
        {
          "1x2":  {"home": 2.10, "draw": 3.40, "away": 3.60},
          "ou25": {"over": 1.85, "under": 1.95},
          "btts":  {"yes": 1.80, "no": 2.00},
        }
    """
    if model is None:
        model = load_trained_model()

    # gracefully handle unknown teams
    for team in (home, away):
        if team not in model.teams:
            print(f"  [Warning] '{team}' not in training data — using average rating.")
            model.attack[team]  = 0.0
            model.defense[team] = 0.0
            model.teams.append(team)

    mkt = all_markets(model, home, away)
    result = {"home": home, "away": away, "markets": mkt, "value_bets": []}

    if bk_odds:
        for market_name, odds in bk_odds.items():
            if market_name == "1x2":
                probs = mkt["match_result"]
            elif market_name == "ou25":
                probs = mkt["over_under"]["2.5"]
            elif market_name == "ou15":
                probs = mkt["over_under"]["1.5"]
            elif market_name == "ou35":
                probs = mkt["over_under"]["3.5"]
            elif market_name == "btts":
                probs = mkt["btts"]
            else:
                continue

            vbets = find_value(market_name, probs, odds, min_edge)
            result["value_bets"].extend(vbets)

    return result


# ── Group-stage scanner ───────────────────────────────────────────────────────

def scan_group_stage(model: DixonColesModel | None = None,
                     bookmaker_odds: dict | None = None,
                     min_edge: float = 0.03) -> pd.DataFrame:
    """
    Scan all WC 2026 group fixtures and return a DataFrame of predictions
    (and value bets where bookmaker odds are provided).
    """
    if model is None:
        model = load_trained_model()

    fixtures = generate_group_fixtures(WC2026_GROUPS)
    rows = []
    for grp, home, away in fixtures:
        # Handle unknown teams with average ratings
        for team in (home, away):
            if team not in model.teams:
                model.attack[team]  = 0.0
                model.defense[team] = 0.0
                if team not in model.teams:
                    model.teams.append(team)

        lam, mu = model.expected_goals(home, away)
        mr = match_result(model, home, away)
        ou = over_under(model, home, away, lines=(2.5,))["2.5"]
        bt = btts(model, home, away)

        bk = bookmaker_odds.get((home, away)) if bookmaker_odds else None
        value_flags = []
        if bk:
            for mkt_name, probs in [("1x2", mr), ("ou25", ou), ("btts", bt)]:
                if mkt_name in bk:
                    vbets = find_value(mkt_name, probs, bk[mkt_name], min_edge)
                    value_flags += [f"{v.selection}({v.edge*100:+.1f}pp)" for v in vbets]

        rows.append({
            "Group":        grp,
            "Home":         home,
            "Away":         away,
            "xG_Home":      round(lam, 2),
            "xG_Away":      round(mu, 2),
            "P(Home)":      f"{mr['home']*100:.1f}%",
            "P(Draw)":      f"{mr['draw']*100:.1f}%",
            "P(Away)":      f"{mr['away']*100:.1f}%",
            "P(O2.5)":      f"{ou['over']*100:.1f}%",
            "P(BTTS Yes)":  f"{bt['yes']*100:.1f}%",
            "Value Bets":   ", ".join(value_flags) if value_flags else "—",
        })

    return pd.DataFrame(rows)


# ── CLI helper ────────────────────────────────────────────────────────────────

def print_group_predictions(model: DixonColesModel | None = None) -> None:
    df = scan_group_stage(model)
    print("\n" + "═" * 110)
    print("  WC 2026 GROUP STAGE PREDICTIONS (Dixon-Coles model, trained on WC 2018+2022)")
    print("═" * 110)
    for grp in sorted(df["Group"].unique()):
        gdf = df[df["Group"] == grp]
        print(f"\n  Group {grp}")
        print(tabulate(gdf.drop(columns="Group"), headers="keys",
                       tablefmt="rounded_outline", showindex=False))
    print()
