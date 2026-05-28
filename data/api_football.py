"""
API-Football client (api-football.com / api-sports.io).

Sign up at https://www.api-football.com/ — Pro plan ~$19/month.
Gives corners, shots, cards, fouls, possession for:
  - All WC qualifier campaigns (all 6 confederations)
  - UEFA Nations League
  - UEFA Euros, Copa America, AFCON, World Cup

Usage:
    from data.api_football import APIFootballClient
    client = APIFootballClient(api_key="YOUR_KEY")
    df = client.fetch_competition_stats(league_id=1, season=2022)   # WC 2022
    df = client.fetch_all_international_stats(seasons=[2022, 2021, 2020])
"""

from __future__ import annotations
import os
import time
import json
import urllib.request
import urllib.parse
import pandas as pd


class APIFootballClient:
    """
    REST client for v3.football.api-sports.io.

    Sign up at https://www.api-football.com/
    Pro plan ~$19/month — 7,500 req/day, all competitions, multi-season history.
    """

    BASE_URL = "https://v3.football.api-sports.io"

    # International competition IDs on API-Football
    COMPETITIONS: dict[str, int] = {
        "FIFA World Cup":          1,
        "WC Qualifiers UEFA":      32,
        "WC Qualifiers CONMEBOL":  31,
        "WC Qualifiers AFC":       30,
        "WC Qualifiers CAF":       29,
        "WC Qualifiers CONCACAF":  34,
        "WC Qualifiers OFC":       35,
        "UEFA Nations League":     5,
        "UEFA Euro":               4,
        "Copa America":            9,
        "Africa Cup of Nations":   6,
    }

    # Stat key → our column name
    _STAT_MAP: dict[str, str] = {
        "Corner Kicks":       "corners",
        "Total Shots":        "shots",
        "Shots on Goal":      "shots_on_target",
        "Yellow Cards":       "yellow",
        "Red Cards":          "red",
        "Fouls":              "fouls",
        "Ball Possession":    "possession_pct",
    }

    def __init__(self, api_key: str, rate_limit_delay: float = 0.5):
        self.api_key = api_key
        self.delay   = rate_limit_delay

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{self.BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={
            "x-apisports-key": self.api_key,
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    def _fetch_fixtures(self, league_id: int, season: int) -> list[dict]:
        """Return list of fixture dicts for a competition/season."""
        data = self._get("fixtures", {"league": league_id, "season": season})
        return data.get("response", [])

    def _fetch_fixture_stats(self, fixture_id: int) -> list[dict]:
        """Return statistics for a single fixture."""
        time.sleep(self.delay)
        data = self._get("fixtures/statistics", {"fixture": fixture_id})
        return data.get("response", [])

    def _parse_stats(self, team_stats: list[dict]) -> dict[str, float]:
        """Parse one team's statistics list into a flat dict."""
        out: dict[str, float] = {}
        for entry in team_stats:
            key = entry.get("type", "")
            val = entry.get("value")
            col = self._STAT_MAP.get(key)
            if col is None or val is None:
                continue
            if isinstance(val, str) and val.endswith("%"):
                val = float(val.rstrip("%"))
            else:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
            out[col] = val
        return out

    def fetch_competition_stats(
        self,
        league_id: int,
        season: int,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all match statistics for a given competition and season.
        Returns a DataFrame with one row per match.
        """
        if verbose:
            comp_name = next((k for k, v in self.COMPETITIONS.items() if v == league_id), str(league_id))
            print(f"  Fetching {comp_name} {season}…", flush=True)

        fixtures = self._fetch_fixtures(league_id, season)
        rows = []

        for i, fix in enumerate(fixtures, 1):
            if verbose:
                print(f"\r  {i}/{len(fixtures)} fixtures…", end="", flush=True)

            fid = fix["fixture"]["id"]
            date = fix["fixture"]["date"][:10]
            home_name = fix["teams"]["home"]["name"]
            away_name = fix["teams"]["away"]["name"]
            home_goals = fix["goals"]["home"]
            away_goals = fix["goals"]["away"]

            # Skip unfinished matches
            if home_goals is None or away_goals is None:
                continue

            stats = self._fetch_fixture_stats(fid)
            if len(stats) < 2:
                continue

            home_stats = next((s for s in stats if s["team"]["name"] == home_name), None)
            away_stats = next((s for s in stats if s["team"]["name"] != home_name), None)

            if not home_stats or not away_stats:
                continue

            hp = self._parse_stats(home_stats.get("statistics", []))
            ap = self._parse_stats(away_stats.get("statistics", []))

            rows.append({
                "league_id":             league_id,
                "season":                season,
                "date":                  date,
                "home_team":             home_name,
                "away_team":             away_name,
                "home_goals":            home_goals,
                "away_goals":            away_goals,
                "home_corners":          hp.get("corners", 0),
                "away_corners":          ap.get("corners", 0),
                "home_shots":            hp.get("shots", 0),
                "away_shots":            ap.get("shots", 0),
                "home_shots_on_target":  hp.get("shots_on_target", 0),
                "away_shots_on_target":  ap.get("shots_on_target", 0),
                "home_yellow":           hp.get("yellow", 0),
                "away_yellow":           ap.get("yellow", 0),
                "home_red":              hp.get("red", 0),
                "away_red":              ap.get("red", 0),
                "home_fouls":            hp.get("fouls", 0),
                "away_fouls":            ap.get("fouls", 0),
                "home_possession_pct":   hp.get("possession_pct", 50.0),
                "away_possession_pct":   ap.get("possession_pct", 50.0),
            })

        if verbose:
            print(f"\r  {len(rows)} matches loaded.            ")

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch_all_international_stats(
        self,
        seasons: list[int] | None = None,
        competitions: list[str] | None = None,
        cache_path: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch stats for all international competitions across multiple seasons.
        Caches results to a CSV to avoid re-fetching.

        seasons     : e.g. [2019, 2020, 2021, 2022, 2023, 2024]
        competitions: list of keys from COMPETITIONS (default: all)
        """
        if seasons is None:
            seasons = list(range(2018, 2026))
        if competitions is None:
            competitions = list(self.COMPETITIONS.keys())

        cache = cache_path or os.path.join(
            os.path.dirname(__file__), "_api_football_stats_cache.csv"
        )

        # Load existing cache
        existing = pd.DataFrame()
        if os.path.exists(cache):
            existing = pd.read_csv(cache, parse_dates=["date"])

        new_rows = []
        for comp_name in competitions:
            league_id = self.COMPETITIONS[comp_name]
            for season in seasons:
                # Skip if already cached
                if not existing.empty:
                    mask = (existing["league_id"] == league_id) & (existing["season"] == season)
                    if mask.any():
                        continue

                try:
                    df_season = self.fetch_competition_stats(league_id, season, verbose=verbose)
                    if not df_season.empty:
                        new_rows.append(df_season)
                except Exception as e:
                    if verbose:
                        print(f"  WARN: {comp_name} {season}: {e}")
                    continue

        if new_rows:
            combined = pd.concat([existing] + new_rows, ignore_index=True)
            combined.sort_values("date", inplace=True)
            combined.to_csv(cache, index=False)
            return combined

        return existing


# ── Normalise team names from API-Football ───────────────────────────────────

from data.loader import TEAM_ALIASES as _GOAL_ALIASES

_API_FOOTBALL_ALIASES: dict[str, str] = {
    **_GOAL_ALIASES,
    "United States":         "USA",
    "Korea Republic":        "South Korea",
    "Korea DPR":             "North Korea",
    "Iran":                  "Iran",
    "Ivory Coast":           "Ivory Coast",
    "Bosnia":                "Bosnia",
    "Bosnia & Herzegovina":  "Bosnia",
    "Czechia":               "Czechia",
    "Czech Republic":        "Czechia",
    "DR Congo":              "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Cape Verde":            "Cape Verde",
    "Guinea-Bissau":         "Guinea-Bissau",
}


def normalise_api_football(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team"] = df["home_team"].replace(_API_FOOTBALL_ALIASES)
    df["away_team"] = df["away_team"].replace(_API_FOOTBALL_ALIASES)
    return df
