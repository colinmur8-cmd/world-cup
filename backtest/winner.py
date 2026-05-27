"""
Winner-prediction backtest.

For each historical WC (2018, 2022):
  1. Download international results up to the tournament start date.
  2. Fit Dixon-Coles on that pre-tournament data.
  3. Simulate the full tournament 50 000 times.
  4. Report: predicted win%, actual winner's rank, top-N accuracy, log-loss.

Then (optionally) predict the 2026 winner using the same methodology.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

import pandas as pd
from tabulate import tabulate

from data.loader import load_full_history
from model.dixon_coles import DixonColesModel
from model.simulate_tournament import simulate_tournament


# ── Historical WC metadata ────────────────────────────────────────────────────

@dataclass
class TournamentMeta:
    year: str
    cutoff_date: str          # day before kick-off
    groups: dict[str, list[str]]
    actual_winner: str
    format32: bool = True     # True = 32-team; False = 48-team


# 2018 groups (Russia)
_GROUPS_2018 = {
    "A": ["Russia",      "Saudi Arabia", "Egypt",        "Uruguay"],
    "B": ["Portugal",    "Spain",        "Morocco",      "Iran"],
    "C": ["France",      "Australia",    "Peru",         "Denmark"],
    "D": ["Argentina",   "Iceland",      "Croatia",      "Nigeria"],
    "E": ["Brazil",      "Switzerland",  "Costa Rica",   "Serbia"],
    "F": ["Germany",     "Mexico",       "Sweden",       "South Korea"],
    "G": ["Belgium",     "Panama",       "Tunisia",      "England"],
    "H": ["Poland",      "Senegal",      "Colombia",     "Japan"],
}

# 2022 groups (Qatar)
_GROUPS_2022 = {
    "A": ["Qatar",         "Ecuador",      "Senegal",      "Netherlands"],
    "B": ["England",       "Iran",         "USA",          "Wales"],
    "C": ["Argentina",     "Saudi Arabia", "Mexico",       "Poland"],
    "D": ["France",        "Australia",    "Denmark",      "Tunisia"],
    "E": ["Spain",         "Costa Rica",   "Germany",      "Japan"],
    "F": ["Belgium",       "Canada",       "Morocco",      "Croatia"],
    "G": ["Brazil",        "Serbia",       "Switzerland",  "Cameroon"],
    "H": ["Portugal",      "Ghana",        "Uruguay",      "South Korea"],
}

WC_HISTORY: list[TournamentMeta] = [
    TournamentMeta(
        year="2018",
        cutoff_date="2018-06-14",
        groups=_GROUPS_2018,
        actual_winner="France",
        format32=True,
    ),
    TournamentMeta(
        year="2022",
        cutoff_date="2022-11-20",
        groups=_GROUPS_2022,
        actual_winner="Argentina",
        format32=True,
    ),
]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class WinnerBacktestResult:
    year: str
    actual_winner: str
    n_sims: int
    n_training_matches: int
    win_probs: dict[str, float]          # team → probability, sorted desc

    @property
    def ranking(self) -> list[tuple[str, float]]:
        """Teams sorted by predicted win probability."""
        return list(self.win_probs.items())

    @property
    def winner_rank(self) -> int:
        teams = list(self.win_probs.keys())
        try:
            return teams.index(self.actual_winner) + 1
        except ValueError:
            return -1

    @property
    def winner_prob(self) -> float:
        return self.win_probs.get(self.actual_winner, 0.0)

    def top_n_accuracy(self, n: int) -> bool:
        return self.winner_rank <= n

    def log_loss(self) -> float:
        eps = 1e-10
        return -math.log(max(self.winner_prob, eps))


# ── Main backtest runner ──────────────────────────────────────────────────────

def run_winner_backtest(
    window_years: int = 8,
    n_sims: int = 50_000,
    seed: int = 42,
    cache_path: str | None = None,
    verbose: bool = True,
) -> list[WinnerBacktestResult]:
    """
    Run the winner-prediction backtest for all tournaments in WC_HISTORY.
    Returns one WinnerBacktestResult per tournament.
    """
    results = []

    for meta in WC_HISTORY:
        if verbose:
            print(f"\n  ── {meta.year} WC backtest ──")
            print(f"     Cutoff: {meta.cutoff_date}  |  window: {window_years} years  |  sims: {n_sims:,}")

        hist = load_full_history(
            cutoff_date=meta.cutoff_date,
            window_years=window_years,
            competitive_only=True,
            cache_path=cache_path,
        )

        if verbose:
            print(f"     Training matches: {len(hist):,}")

        model = DixonColesModel().fit(hist)

        probs = simulate_tournament(
            meta.groups,
            model,
            n_sims=n_sims,
            seed=seed,
            format32=meta.format32,
        )

        res = WinnerBacktestResult(
            year=meta.year,
            actual_winner=meta.actual_winner,
            n_sims=n_sims,
            n_training_matches=len(hist),
            win_probs=probs,
        )
        results.append(res)

        if verbose:
            _print_single(res, top_n=10)

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_single(res: WinnerBacktestResult, top_n: int = 10) -> None:
    rows = []
    for rank, (team, prob) in enumerate(res.ranking[:top_n], 1):
        marker = " ← ACTUAL WINNER ✓" if team == res.actual_winner else ""
        rows.append((rank, team, f"{prob*100:.1f}%", marker))

    # Add winner row if outside top_n
    if res.winner_rank > top_n:
        rows.append(("…", "", "", ""))
        rows.append((
            res.winner_rank,
            res.actual_winner,
            f"{res.winner_prob*100:.1f}%",
            " ← ACTUAL WINNER",
        ))

    print()
    print(tabulate(rows, headers=["Rank", "Team", "Win Prob", ""],
                   tablefmt="simple", colalign=("right", "left", "right", "left")))
    print(f"\n     Winner rank: {res.winner_rank} / {len(res.win_probs)}")
    print(f"     Winner predicted prob: {res.winner_prob*100:.1f}%")
    print(f"     Top-1: {'✓' if res.top_n_accuracy(1) else '✗'}  "
          f"Top-3: {'✓' if res.top_n_accuracy(3) else '✗'}  "
          f"Top-5: {'✓' if res.top_n_accuracy(5) else '✗'}  "
          f"Top-8: {'✓' if res.top_n_accuracy(8) else '✗'}")


def print_backtest_summary(results: list[WinnerBacktestResult]) -> None:
    sep = "═" * 68
    print(f"\n{sep}")
    print("  WC WINNER PREDICTION — BACKTEST SUMMARY")
    print(f"{sep}")

    summary_rows = []
    for res in results:
        summary_rows.append((
            res.year,
            res.actual_winner,
            f"{res.winner_prob*100:.1f}%",
            f"{res.winner_rank} / {len(res.win_probs)}",
            "✓" if res.top_n_accuracy(1) else "✗",
            "✓" if res.top_n_accuracy(3) else "✗",
            "✓" if res.top_n_accuracy(5) else "✗",
            f"{res.log_loss():.3f}",
        ))

    print()
    print(tabulate(
        summary_rows,
        headers=["WC", "Winner", "Pred. Prob", "Rank", "Top-1", "Top-3", "Top-5", "LogLoss"],
        tablefmt="rounded_outline",
        colalign=("left","left","right","right","center","center","center","right"),
    ))
    print()
