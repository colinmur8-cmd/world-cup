"""
Live odds fetcher via The Odds API (https://the-odds-api.com).
Free tier: 500 requests/month.  Register at the-odds-api.com to get a key.

Usage:
    from data.odds_fetcher import fetch_wc_odds
    odds = fetch_wc_odds("YOUR_API_KEY")
    # odds: dict keyed by (home_team, away_team) → {"1x2": {...}, "ou25": {...}, "btts": {...}}
"""
from __future__ import annotations
import json, urllib.request
from data.loader import TEAM_ALIASES

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"

# Extra name mappings for odds-API → our canonical names
_ODDS_ALIASES: dict[str, str] = {
    "United States":              "USA",
    "Bosnia and Herzegovina":     "Bosnia",
    "Cote d'Ivoire":              "Ivory Coast",
    "Côte d'Ivoire":              "Ivory Coast",
    "DR Congo":                   "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Cape Verde Islands":         "Cape Verde",
    "Curaçao":                    "Curacao",
    "Czech Republic":             "Czechia",
    "Republic of Ireland":        "Ireland",
    "Korea Republic":             "South Korea",
    "South Korea":                "South Korea",
    "IR Iran":                    "Iran",
}

def _normalise(name: str) -> str:
    name = name.strip()
    return _ODDS_ALIASES.get(name, TEAM_ALIASES.get(name, name))

def fetch_wc_odds(
    api_key: str,
    bookmakers: str = "bet365,pinnacle,betfair_ex_eu",
    regions: str = "eu",
    markets: str = "h2h,totals,btts",
) -> dict[tuple[str, str], dict]:
    """
    Fetch current WC 2026 match odds.

    Returns a dict keyed by (home_team, away_team) tuples (canonical names).
    Each value is a dict with keys "1x2", "ou25", "btts" where available.

    The returned odds are the best available across the requested bookmakers.
    """
    url = (
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds/"
        f"?apiKey={api_key}"
        f"&regions={regions}"
        f"&bookmakers={bookmakers}"
        f"&markets={markets}"
        f"&oddsFormat=decimal"
    )

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Odds API error {e.code}: {body}") from e

    result: dict[tuple[str, str], dict] = {}

    for event in data:
        home = _normalise(event.get("home_team", ""))
        away = _normalise(event.get("away_team", ""))
        if not home or not away:
            continue

        best: dict[str, dict] = {}

        for bk in event.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                key = mkt["key"]
                outcomes = mkt.get("outcomes", [])

                if key == "h2h":
                    row: dict[str, float] = {}
                    for o in outcomes:
                        name = _normalise(o["name"])
                        if name == home:
                            row["home"] = o["price"]
                        elif name == away:
                            row["away"] = o["price"]
                        elif o["name"].lower() == "draw":
                            row["draw"] = o["price"]
                    if len(row) == 3:
                        # Take best (highest) odds for each selection
                        prev = best.get("1x2", {})
                        best["1x2"] = {
                            k: max(row.get(k, 1.0), prev.get(k, 1.0))
                            for k in ("home", "draw", "away")
                        }

                elif key == "totals":
                    for line_str in ("2.5", "1.5", "3.5"):
                        line_outcomes = [o for o in outcomes
                                         if abs(float(o.get("point", 0)) - float(line_str)) < 0.01]
                        if len(line_outcomes) == 2:
                            mkt_name = f"ou{line_str.replace('.','')}"
                            over_o  = next((o for o in line_outcomes if o["name"].lower() == "over"),  None)
                            under_o = next((o for o in line_outcomes if o["name"].lower() == "under"), None)
                            if over_o and under_o:
                                prev = best.get(mkt_name, {})
                                best[mkt_name] = {
                                    "over":  max(over_o["price"],  prev.get("over",  1.0)),
                                    "under": max(under_o["price"], prev.get("under", 1.0)),
                                }

                elif key == "btts":
                    yes_o = next((o for o in outcomes if o["name"].lower() in ("yes", "both teams to score - yes")), None)
                    no_o  = next((o for o in outcomes if o["name"].lower() in ("no",  "both teams to score - no")),  None)
                    if yes_o and no_o:
                        prev = best.get("btts", {})
                        best["btts"] = {
                            "yes": max(yes_o["price"], prev.get("yes", 1.0)),
                            "no":  max(no_o["price"],  prev.get("no",  1.0)),
                        }

        if best:
            result[(home, away)] = best

    return result


def print_available_odds(api_key: str) -> None:
    """Pretty-print all available WC matches with their current odds."""
    from tabulate import tabulate
    odds = fetch_wc_odds(api_key)
    if not odds:
        print("  No odds found — tournament may not have opened yet, or check your API key.")
        return

    rows = []
    for (home, away), mkts in sorted(odds.items()):
        mr = mkts.get("1x2", {})
        ou = mkts.get("ou25", {})
        rows.append([
            home, away,
            f"{mr.get('home','—'):.2f}" if mr.get('home') else "—",
            f"{mr.get('draw','—'):.2f}" if mr.get('draw') else "—",
            f"{mr.get('away','—'):.2f}" if mr.get('away') else "—",
            f"{ou.get('over','—'):.2f}" if ou.get('over') else "—",
            f"{ou.get('under','—'):.2f}" if ou.get('under') else "—",
        ])
    print(tabulate(rows,
                   headers=["Home","Away","1 (Home)","X (Draw)","2 (Away)","O2.5","U2.5"],
                   tablefmt="rounded_outline"))
