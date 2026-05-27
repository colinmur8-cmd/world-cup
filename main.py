#!/usr/bin/env python3
"""
WC Betting Model — main entry point.

Usage:
    python main.py                   # full run: backtest + 2026 predictions
    python main.py --backtest-only   # skip 2026 predictions
    python main.py --predict-only    # skip backtest
    python main.py --match "France vs Morocco"   # single match prediction
    python main.py --ratings         # print team attack/defense ratings
"""

import argparse
import sys
import os

# ── Dependency check ─────────────────────────────────────────────────────────
_MISSING = []
for pkg in ("numpy", "scipy", "pandas", "tabulate"):
    try:
        __import__(pkg)
    except ImportError:
        _MISSING.append(pkg)

if _MISSING:
    print(f"Missing packages: {', '.join(_MISSING)}")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
from tabulate import tabulate

from data.loader import load_matches
from model.dixon_coles import DixonColesModel
from model.markets import all_markets
from model.edge import find_value
from backtest.engine import run_backtest
from backtest.report import print_report, save_equity_curve
from predictions.wc2026 import (
    load_trained_model, predict_match, print_group_predictions
)


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="WC 2026 Betting Model")
    p.add_argument("--backtest-only", action="store_true")
    p.add_argument("--predict-only",  action="store_true")
    p.add_argument("--ratings",       action="store_true",
                   help="Print team attack/defense ratings and exit")
    p.add_argument("--match",         type=str, default=None,
                   help='Single match: "Home vs Away"')
    p.add_argument("--no-chart",      action="store_true",
                   help="Skip saving the equity curve chart")
    return p.parse_args()


# ── Sections ──────────────────────────────────────────────────────────────────

def run_ratings(model: DixonColesModel) -> None:
    df = model.team_ratings()
    df["attack"]  = df["attack"].map(lambda x: f"{x:+.3f}")
    df["defense"] = df["defense"].map(lambda x: f"{x:+.3f}")
    print("\n  Team Ratings (Dixon-Coles, trained on WC 2018 + 2022)")
    print("  attack > 0 → above-average scorer  |  defense < 0 → above-average defence\n")
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))


def run_single_match(home: str, away: str, model: DixonColesModel) -> None:
    pred = predict_match(home, away, model)
    mkt  = pred["markets"]
    lam  = mkt["expected_goals"]["home"]
    mu   = mkt["expected_goals"]["away"]
    mr   = mkt["match_result"]
    ou   = mkt["over_under"]
    bt   = mkt["btts"]
    cs   = mkt["correct_score"][:10]

    print(f"\n  {'─'*54}")
    print(f"  {home}  vs  {away}")
    print(f"  Expected goals: {home} {lam:.2f}  —  {away} {mu:.2f}")
    print(f"  {'─'*54}")

    # 1X2
    mr_rows = [("Home win", f"{mr['home']*100:.1f}%",
                f"{1/mr['home']:.2f}"),
               ("Draw",     f"{mr['draw']*100:.1f}%",
                f"{1/mr['draw']:.2f}"),
               ("Away win", f"{mr['away']*100:.1f}%",
                f"{1/mr['away']:.2f}")]
    print(tabulate(mr_rows, headers=["Outcome","Model Prob","Fair Odds"],
                   tablefmt="simple", colalign=("left","right","right")))

    # O/U
    ou_rows = []
    for line in ["1.5","2.5","3.5","4.5"]:
        if line in ou:
            ou_rows.append((f"Over  {line}",
                            f"{ou[line]['over']*100:.1f}%",
                            f"{1/ou[line]['over']:.2f}"))
            ou_rows.append((f"Under {line}",
                            f"{ou[line]['under']*100:.1f}%",
                            f"{1/ou[line]['under']:.2f}"))
    print()
    print(tabulate(ou_rows, headers=["Market","Model Prob","Fair Odds"],
                   tablefmt="simple", colalign=("left","right","right")))

    # BTTS
    btts_rows = [("BTTS Yes", f"{bt['yes']*100:.1f}%", f"{1/bt['yes']:.2f}"),
                 ("BTTS No",  f"{bt['no']*100:.1f}%",  f"{1/bt['no']:.2f}")]
    print()
    print(tabulate(btts_rows, headers=["Market","Model Prob","Fair Odds"],
                   tablefmt="simple", colalign=("left","right","right")))

    # Top scorelines
    print("\n  Top scorelines:")
    cs_rows = [(s, f"{p*100:.2f}%", f"{1/p:.1f}") for s, p in cs]
    print(tabulate(cs_rows, headers=["Score","Prob","Fair Odds"],
                   tablefmt="simple", colalign=("left","right","right")))
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Always train model (fast, < 2s)
    print("  Loading data and fitting Dixon-Coles model (WC 2018 + 2022)…")
    model = load_trained_model()
    print(f"  Model ready — {len(model.teams)} teams, ρ={model.rho:.4f}\n")

    # ── Ratings ──
    if args.ratings:
        run_ratings(model)
        return

    # ── Single match ──
    if args.match:
        parts = [p.strip() for p in args.match.split("vs")]
        if len(parts) != 2:
            print("  Usage: --match \"Team A vs Team B\"")
            sys.exit(1)
        run_single_match(parts[0], parts[1], model)
        return

    # ── Backtest ──
    if not args.predict_only:
        print("  Running backtest (train: WC 2018 → test: WC 2022)…")
        result = run_backtest(train_years=("2018",), test_years=("2022",))
        print_report(result)
        if not args.no_chart:
            save_equity_curve(result)

    # ── 2026 predictions ──
    if not args.backtest_only:
        print_group_predictions(model)


if __name__ == "__main__":
    main()
