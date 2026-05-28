"""
Match-level backtest for WC 2018 and 2022.

For each match:
  1. Train Dixon-Coles on all pre-tournament international results.
  2. Predict 1X2, BTTS, O/U 2.5 probabilities.
  3. Compare to actual result.

Outputs:
  - Per-match prediction accuracy
  - Calibration (do our 65% calls happen 65% of the time?)
  - Brier score and log-loss
  - Simulated P&L against synthetic bookmaker odds (6% margin)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from data.loader import load_full_history, load_matches, TEAM_ALIASES
from model.dixon_coles import DixonColesModel
from model.markets import all_markets
from model.edge import simulate_bookmaker_odds, kelly, find_value


# ── Per-match result container ────────────────────────────────────────────────

@dataclass
class MatchPred:
    year: str
    stage: str
    date: str
    home: str
    away: str
    actual_hg: int
    actual_ag: int
    # Model outputs
    p_home: float
    p_draw: float
    p_away: float
    p_btts: float
    p_over25: float
    xg_home: float
    xg_away: float

    # ── Derived ──

    @property
    def actual_result(self) -> Literal["home", "draw", "away"]:
        if self.actual_hg > self.actual_ag:
            return "home"
        if self.actual_hg < self.actual_ag:
            return "away"
        return "draw"

    @property
    def predicted_result(self) -> Literal["home", "draw", "away"]:
        return max(
            {"home": self.p_home, "draw": self.p_draw, "away": self.p_away},
            key=lambda k: {"home": self.p_home, "draw": self.p_draw, "away": self.p_away}[k],
        )

    @property
    def correct_result(self) -> bool:
        return self.predicted_result == self.actual_result

    @property
    def brier_1x2(self) -> float:
        a = {"home": 0.0, "draw": 0.0, "away": 0.0}
        a[self.actual_result] = 1.0
        return ((self.p_home - a["home"])**2
                + (self.p_draw - a["draw"])**2
                + (self.p_away - a["away"])**2) / 3.0

    @property
    def log_loss_1x2(self) -> float:
        p = {"home": self.p_home, "draw": self.p_draw, "away": self.p_away}
        return -math.log(max(p[self.actual_result], 1e-10))

    @property
    def btts_actual(self) -> bool:
        return self.actual_hg > 0 and self.actual_ag > 0

    @property
    def over25_actual(self) -> bool:
        return (self.actual_hg + self.actual_ag) > 2

    @property
    def brier_btts(self) -> float:
        a = 1.0 if self.btts_actual else 0.0
        return (self.p_btts - a)**2

    @property
    def brier_ou25(self) -> float:
        a = 1.0 if self.over25_actual else 0.0
        return (self.p_over25 - a)**2


# ── Cutoff dates per tournament ───────────────────────────────────────────────

_CUTOFFS = {
    "2018": "2018-06-14",
    "2022": "2022-11-20",
}


# ── Main runner ───────────────────────────────────────────────────────────────

def run_match_backtest(
    years: tuple[str, ...] = ("2018", "2022"),
    window_years: int = 8,
    decay: float = 0.3,
    verbose: bool = False,
) -> list[MatchPred]:
    """
    Train on pre-tournament history, predict every WC match, return results.
    """
    preds: list[MatchPred] = []

    for year in years:
        cutoff = _CUTOFFS[year]
        if verbose:
            print(f"  [{year}] loading history (cutoff {cutoff}, {window_years}y window)…")

        hist = load_full_history(
            cutoff_date=cutoff,
            window_years=window_years,
            competitive_only=True,
        )
        if verbose:
            print(f"  [{year}] {len(hist):,} training matches — fitting model…")

        model = DixonColesModel().fit(hist, decay=decay)

        wc = load_matches((year,))

        for _, row in wc.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            # Add unknown teams with average ratings
            for team in (home, away):
                if team not in model.teams:
                    model.attack[team]  = 0.0
                    model.defense[team] = 0.0
                    model.teams.append(team)

            mkt = all_markets(model, home, away)
            mr  = mkt["match_result"]
            ou  = mkt["over_under"]["2.5"]
            bt  = mkt["btts"]
            xg  = mkt["expected_goals"]

            preds.append(MatchPred(
                year=year,
                stage=row["stage"],
                date=str(row["date"].date()),
                home=home,
                away=away,
                actual_hg=int(row["home_goals"]),
                actual_ag=int(row["away_goals"]),
                p_home=mr["home"],
                p_draw=mr["draw"],
                p_away=mr["away"],
                p_btts=bt["yes"],
                p_over25=ou["over"],
                xg_home=xg["home"],
                xg_away=xg["away"],
            ))

    return preds


# ── Calibration ───────────────────────────────────────────────────────────────

def calibration_table(preds: list[MatchPred], n_bins: int = 10) -> pd.DataFrame:
    """
    Bin all 1X2 probability-outcome pairs into deciles.
    Returns: predicted midpoint, actual frequency, count.
    """
    pairs: list[tuple[float, int]] = []
    for p in preds:
        pairs += [
            (p.p_home, 1 if p.actual_result == "home" else 0),
            (p.p_draw, 1 if p.actual_result == "draw" else 0),
            (p.p_away, 1 if p.actual_result == "away" else 0),
        ]
    pairs.sort(key=lambda x: x[0])

    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [(pr, ac) for pr, ac in pairs if lo <= pr < hi]
        if not bucket:
            continue
        probs, actuals = zip(*bucket)
        rows.append({
            "Predicted (mid)": round((lo + hi) / 2, 2),
            "Actual freq":     round(sum(actuals) / len(actuals), 3),
            "Count":           len(bucket),
        })
    return pd.DataFrame(rows)


# ── P&L simulation ────────────────────────────────────────────────────────────

def pnl_simulation(
    preds: list[MatchPred],
    margin: float = 0.06,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.5,
    flat_stake: float = 1.0,
    bankroll: float = 1000.0,
) -> pd.DataFrame:
    """
    Simulate betting on every match where model edge >= min_edge,
    against synthetic bookmaker odds (model fair odds + margin).

    Returns a DataFrame with one row per bet placed.
    """
    rows = []
    bk_balance = bankroll
    kelly_balance = bankroll

    for p in preds:
        mr_probs = {"home": p.p_home, "draw": p.p_draw, "away": p.p_away}
        bk_odds  = simulate_bookmaker_odds(mr_probs, margin=margin)
        vbets    = find_value("1x2", mr_probs, bk_odds, min_edge=min_edge)

        for vb in vbets:
            won = (vb.selection == p.actual_result)
            payout = vb.bk_odds if won else 0.0

            # Flat stake
            flat_pnl = flat_stake * (payout - 1.0)

            # Kelly stake (fraction of current Kelly balance)
            k_stake  = kelly_fraction * kelly(vb.model_prob, vb.bk_odds) * kelly_balance
            k_pnl    = k_stake * (payout - 1.0)

            bk_balance    += flat_pnl
            kelly_balance += k_pnl

            rows.append({
                "Year":         p.year,
                "Stage":        p.stage,
                "Match":        f"{p.home} vs {p.away}",
                "Selection":    vb.selection,
                "Model %":      f"{vb.model_prob*100:.1f}%",
                "Implied %":    f"{vb.implied_prob*100:.1f}%",
                "Edge":         f"{vb.edge*100:+.1f}pp",
                "Bk Odds":      round(vb.bk_odds, 2),
                "Won":          "✓" if won else "✗",
                "Flat P&L":     round(flat_pnl, 2),
                "Flat Balance": round(bk_balance, 2),
                "Kelly Stake":  round(k_stake, 2),
                "Kelly P&L":    round(k_pnl, 2),
                "Kelly Balance":round(kelly_balance, 2),
            })

    return pd.DataFrame(rows)


# ── Summary metrics ───────────────────────────────────────────────────────────

def summary_metrics(preds: list[MatchPred]) -> dict:
    n = len(preds)
    correct = sum(p.correct_result for p in preds)
    avg_brier = sum(p.brier_1x2 for p in preds) / n
    avg_ll    = sum(p.log_loss_1x2 for p in preds) / n

    btts_correct = sum(
        (p.p_btts >= 0.5) == p.btts_actual for p in preds
    )
    ou_correct = sum(
        (p.p_over25 >= 0.5) == p.over25_actual for p in preds
    )

    return {
        "matches":       n,
        "result_acc":    correct / n,
        "brier_1x2":     avg_brier,
        "log_loss_1x2":  avg_ll,
        "btts_acc":      btts_correct / n,
        "ou25_acc":      ou_correct / n,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tabulate import tabulate

    print("Running match-level backtest (2018 + 2022)…")
    preds = run_match_backtest(verbose=True)

    metrics = summary_metrics(preds)
    print(f"\n  Matches:          {metrics['matches']}")
    print(f"  Result accuracy:  {metrics['result_acc']*100:.1f}%  (baseline naive: ~45%)")
    print(f"  Brier score 1X2:  {metrics['brier_1x2']:.4f}  (lower = better, random = 0.222)")
    print(f"  Log-loss 1X2:     {metrics['log_loss_1x2']:.4f}")
    print(f"  BTTS accuracy:    {metrics['btts_acc']*100:.1f}%")
    print(f"  O/U 2.5 accuracy: {metrics['ou25_acc']*100:.1f}%")

    print("\n  Calibration (1X2 predictions binned by decile):")
    cal = calibration_table(preds)
    print(tabulate(cal, headers="keys", tablefmt="rounded_outline",
                   showindex=False, floatfmt=".3f"))

    print("\n  P&L simulation (6% bookie margin, ≥3pp edge, half-Kelly, £1000 bankroll):")
    pnl = pnl_simulation(preds)
    if pnl.empty:
        print("  No bets found at this edge threshold.")
    else:
        print(f"  Bets placed: {len(pnl)}")
        final_flat   = pnl["Flat Balance"].iloc[-1]
        final_kelly  = pnl["Kelly Balance"].iloc[-1]
        print(f"  Final flat balance:   £{final_flat:.2f}  (started £1000)")
        print(f"  Final Kelly balance:  £{final_kelly:.2f}  (started £1000)")
        print()
        print(tabulate(pnl[["Year","Match","Selection","Model %","Edge","Bk Odds","Won","Flat P&L","Kelly P&L"]],
                       headers="keys", tablefmt="simple", showindex=False))
