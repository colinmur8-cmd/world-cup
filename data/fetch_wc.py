"""
Fetch verified WC match data from public sources and save as a local CSV.

Sources tried in order:
  1. GitHub: martj42/international-football-results (comprehensive, free)
  2. football-data.org API (requires free API key for competition-level data)
  3. Local file fallback: any CSV in the ./data/ directory named results*.csv

Outputs: data/wc_verified.csv  (same schema as load_from_csv expects)

Usage:
    python -m data.fetch_wc                          # auto-detect source
    python -m data.fetch_wc --api-key YOUR_KEY       # football-data.org
    python -m data.fetch_wc --local data/results.csv # use local CSV
"""

from __future__ import annotations
import os
import sys
import argparse
import pandas as pd

OUT_PATH = os.path.join(os.path.dirname(__file__), "wc_verified.csv")

GITHUB_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/results.csv"
)

FD_ORG_URLS = {
    "2018": "https://api.football-data.org/v4/competitions/WC/matches?season=2018",
    "2022": "https://api.football-data.org/v4/competitions/WC/matches?season=2022",
}


# ── Source 1: GitHub CSV ──────────────────────────────────────────────────────

def _fetch_github() -> pd.DataFrame | None:
    try:
        import urllib.request, io
        print("  Trying GitHub (martj42/international-football-results)…")
        with urllib.request.urlopen(GITHUB_URL, timeout=10) as r:
            raw = r.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        df["date"] = pd.to_datetime(df["date"])
        wc = df[
            df["tournament"].str.upper().str.contains("FIFA WORLD CUP", na=False) &
            df["date"].dt.year.isin([2018, 2022])
        ].copy()
        print(f"  Found {len(wc)} WC 2018+2022 matches via GitHub.")
        return wc[["date","home_team","away_team","home_score","away_score","tournament"]]
    except Exception as e:
        print(f"  GitHub source failed: {e}")
        return None


# ── Source 2: football-data.org API ──────────────────────────────────────────

def _fetch_fdorg(api_key: str) -> pd.DataFrame | None:
    try:
        import urllib.request, json
        rows = []
        for season, url in FD_ORG_URLS.items():
            print(f"  Trying football-data.org (WC {season})…")
            req = urllib.request.Request(url, headers={"X-Auth-Token": api_key})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            for m in data.get("matches", []):
                if m["status"] != "FINISHED":
                    continue
                rows.append({
                    "date":      m["utcDate"][:10],
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                    "home_score": m["score"]["fullTime"]["home"],
                    "away_score": m["score"]["fullTime"]["away"],
                    "tournament": f"FIFA World Cup {season}",
                })
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        print(f"  Found {len(df)} matches via football-data.org.")
        return df
    except Exception as e:
        print(f"  football-data.org source failed: {e}")
        return None


# ── Source 3: local CSV ───────────────────────────────────────────────────────

def _load_local(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  Local file not found: {path}")
        return None
    print(f"  Loading local CSV: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    # Filter to WC if tournament column exists
    if "tournament" in df.columns:
        df = df[
            df["tournament"].str.upper().str.contains("WORLD CUP", na=False) &
            df["date"].dt.year.isin([2018, 2022])
        ].copy()
    else:
        df = df[df["date"].dt.year.isin([2018, 2022])].copy()
    print(f"  Found {len(df)} matches in local CSV.")
    return df


# ── Save + normalise ──────────────────────────────────────────────────────────

def _save(df: pd.DataFrame) -> None:
    rename = {
        "home_score": "home_goals", "away_score": "away_goals",
        "score1": "home_goals",     "score2": "away_goals",
    }
    df = df.rename(columns=rename)
    required = {"date","home_team","away_team","home_goals","away_goals"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Cannot save: missing columns {missing}")
    df["tournament"] = df["date"].dt.year.astype(str)
    out = df[["tournament","date","home_team","away_team","home_goals","away_goals"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(out)} matches → {OUT_PATH}")
    print("  Run 'python -m data.validate --csv data/wc_verified.csv' to check for diffs.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=None,
                    help="football-data.org API key (free tier)")
    ap.add_argument("--local", default=None,
                    help="Path to a local CSV file (Kaggle, manual export, etc.)")
    args = ap.parse_args()

    df: pd.DataFrame | None = None

    if args.local:
        df = _load_local(args.local)
    elif args.api_key:
        df = _fetch_fdorg(args.api_key)
    else:
        df = _fetch_github()
        if df is None and args.api_key:
            df = _fetch_fdorg(args.api_key)

    if df is None or df.empty:
        print("\n  No data retrieved. Options:")
        print("    1. Run with --local <path> pointing to a downloaded CSV")
        print("    2. Run with --api-key <key> for football-data.org")
        print("    3. Download https://www.kaggle.com/datasets/martj42/international-football-results")
        print("       and run: python -m data.fetch_wc --local results.csv")
        sys.exit(1)

    _save(df)


if __name__ == "__main__":
    main()
