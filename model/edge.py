"""
Edge and value calculation utilities.

Edge = model probability - implied bookmaker probability.
A positive edge means our model thinks the outcome is more likely
than the market implies — this is where value lives.

Kelly criterion staking is included for bankroll management.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class ValueBet:
    market: str
    selection: str
    model_prob: float
    bk_odds: float          # decimal odds (European format)
    implied_prob: float
    edge: float             # model_prob - implied_prob
    kelly_fraction: float   # full Kelly stake as fraction of bankroll
    half_kelly: float       # recommended: half Kelly


def decimal_to_implied(odds: float) -> float:
    """Convert decimal odds → implied probability (no margin removed)."""
    return 1.0 / odds


def remove_margin(odds_dict: dict[str, float]) -> dict[str, float]:
    """
    Remove bookmaker overround from a set of decimal odds.
    Returns fair implied probabilities that sum to 1.0.
    """
    raw = {k: 1.0 / v for k, v in odds_dict.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def kelly(prob: float, odds: float) -> float:
    """
    Full Kelly fraction: f* = (p*b - q) / b
    where b = decimal_odds - 1, q = 1 - p.
    Returns 0 if no edge.
    """
    b = odds - 1.0
    q = 1.0 - prob
    f = (prob * b - q) / b
    return max(f, 0.0)


def find_value(market: str, model_probs: dict[str, float],
               bk_odds: dict[str, float],
               min_edge: float = 0.03) -> list[ValueBet]:
    """
    Compare model probabilities against bookmaker odds.

    Returns ValueBet objects for any selection where
    edge >= min_edge (default 3 percentage points).
    """
    bets: list[ValueBet] = []
    fair_implied = remove_margin(bk_odds)

    for selection, mp in model_probs.items():
        if selection not in bk_odds:
            continue
        odds   = bk_odds[selection]
        imp    = fair_implied[selection]
        edge   = mp - imp
        if edge >= min_edge:
            kf = kelly(mp, odds)
            bets.append(ValueBet(
                market=market,
                selection=selection,
                model_prob=round(mp, 4),
                bk_odds=odds,
                implied_prob=round(imp, 4),
                edge=round(edge, 4),
                kelly_fraction=round(kf, 4),
                half_kelly=round(kf / 2, 4),
            ))
    return sorted(bets, key=lambda b: -b.edge)


def simulate_bookmaker_odds(true_probs: dict[str, float],
                            margin: float = 0.06) -> dict[str, float]:
    """
    Synthesise bookmaker odds by inflating probabilities by a margin,
    then converting to decimal odds.
    Used for backtesting when historical odds are unavailable.
    """
    total = sum(true_probs.values())
    normed = {k: v / total for k, v in true_probs.items()}
    # Inflate each implied prob by the margin uniformly
    inflated = {k: v * (1 + margin) for k, v in normed.items()}
    # Back to decimal odds
    return {k: round(1.0 / v, 3) for k, v in inflated.items()}


def expected_value(prob: float, odds: float) -> float:
    """EV = prob * (odds - 1) - (1 - prob). Positive → profitable bet."""
    return prob * (odds - 1.0) - (1.0 - prob)
