"""
Backtesting engine.

Methodology:
  - Train Dixon-Coles on WC 2018 matches.
  - For each WC 2022 match, generate model probabilities.
  - Synthetic bookmaker odds are derived from a NAIVE PRIOR (historical WC
    base rates), NOT from the test model — avoiding circular validation.
  - Bet where model edge vs. naive prior >= MIN_EDGE using half-Kelly stakes.
  - Record P&L, ROI, Sharpe, max drawdown per market.
  - Calibration: compare model predicted probabilities to actual outcome rates.

Naive prior baseline (WC neutral-venue historical base rates ≈ 2010-2022):
  1X2:  Home 38% / Draw 27% / Away 35%
  O/U2.5: Over 50% / Under 50%
  BTTS:  Yes 44% / No 56%
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from data.loader import load_matches
from model.dixon_coles import DixonColesModel
from model.markets import match_result, over_under, btts
from model.edge import find_value, ValueBet


MIN_EDGE           = 0.05    # minimum edge to bet (5 pp vs naive prior)
BK_MARGIN          = 0.055   # simulated bookmaker overround (5.5%)
BANKROLL           = 1000.0
MAX_STAKE_FRACTION = 0.03    # cap each bet at 3% of current bankroll (drawdown control)
SHRINK_ALPHA       = 0.15    # shrink model probs 15% toward flat prior (reduces overconfidence)

# ── Naive bookmaker prior (non-circular baseline) ─────────────────────────────
# Approximate WC neutral-ground base rates used to generate synthetic odds.
_PRIOR = {
    "1x2":  {"home": 0.38, "draw": 0.27, "away": 0.35},
    "ou25": {"over": 0.50, "under": 0.50},
    "btts":  {"yes": 0.44, "no":  0.56},
}

def _prior_odds(market: str) -> dict[str, float]:
    """Convert naive prior probabilities to decimal odds with bookmaker margin."""
    probs = _PRIOR[market]
    return {k: round(1.0 / (p * (1 + BK_MARGIN)), 3) for k, p in probs.items()}

def _shrink(probs: dict[str, float], market: str) -> dict[str, float]:
    """Mix model probs toward flat prior to reduce overconfidence on limited data."""
    prior = _PRIOR.get(market, {})
    out = {}
    for k, p in probs.items():
        pp = prior.get(k, 1.0 / len(probs))
        out[k] = (1 - SHRINK_ALPHA) * p + SHRINK_ALPHA * pp
    return out


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BetRecord:
    match:      str
    date:       str
    market:     str
    selection:  str
    model_prob: float
    prior_prob: float
    bk_odds:    float
    edge:       float           # model_prob - implied prior prob (after margin)
    stake:      float
    result:     str             # "win" | "loss"
    pnl:        float


@dataclass
class CalibrationBucket:
    bin_centre: float
    predicted:  float
    actual:     float
    count:      int


@dataclass
class BacktestResult:
    bets:             list[BetRecord]         = field(default_factory=list)
    bankroll_curve:   list[float]             = field(default_factory=list)
    # Calibration data: list of (predicted_prob, outcome 0/1) for each market
    calib_records:    list[tuple[str,float,int]] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.bets:
            return {"error": "no bets placed"}

        df = pd.DataFrame([vars(b) for b in self.bets])
        total_staked = df["stake"].sum()
        total_pnl    = df["pnl"].sum()
        roi          = total_pnl / total_staked if total_staked > 0 else 0.0
        wins         = (df["result"] == "win").sum()
        win_rate     = wins / len(df)

        pnls   = df["pnl"].values
        sharpe = (pnls.mean() / pnls.std(ddof=1)) * math.sqrt(len(pnls)) if len(pnls) > 1 else 0.0

        curve  = np.array(self.bankroll_curve)
        peaks  = np.maximum.accumulate(curve)
        dd     = (curve - peaks) / peaks
        max_dd = float(dd.min())

        by_market = (df.groupby("market")
                       .agg(bets=("pnl","count"),
                            wins=("result", lambda x: (x=="win").sum()),
                            staked=("stake","sum"),
                            pnl=("pnl","sum"))
                       .assign(roi=lambda d: d["pnl"]/d["staked"])
                       .reset_index())

        return {
            "total_bets":    len(df),
            "total_wins":    int(wins),
            "win_rate":      round(win_rate, 4),
            "total_staked":  round(total_staked, 2),
            "total_pnl":     round(total_pnl, 2),
            "roi":           round(roi, 4),
            "sharpe":        round(sharpe, 4),
            "max_drawdown":  round(max_dd, 4),
            "by_market":     by_market,
            "final_bankroll": round(self.bankroll_curve[-1], 2),
        }

    def calibration_report(self, n_bins: int = 5) -> list[CalibrationBucket]:
        """Bin predicted probabilities and compare to observed frequencies."""
        if not self.calib_records:
            return []
        recs = sorted(self.calib_records, key=lambda x: x[1])
        bins = np.linspace(0, 1, n_bins + 1)
        buckets = []
        for lo, hi in zip(bins, bins[1:]):
            group = [(m, p, o) for m, p, o in recs if lo <= p < hi]
            if len(group) >= 3:
                preds = [p for _, p, _ in group]
                outcomes = [o for _, _, o in group]
                buckets.append(CalibrationBucket(
                    bin_centre=round((lo + hi) / 2, 2),
                    predicted=round(sum(preds) / len(preds), 3),
                    actual=round(sum(outcomes) / len(outcomes), 3),
                    count=len(group),
                ))
        return buckets

    def log_loss(self) -> tuple[float, float]:
        """
        Multinomial log-loss per match for 1X2 market only.
        calib_records stores (market, pred_prob_of_outcome, outcome_happened).
        We filter to 1x2 home records which capture the full 3-way distribution.
        """
        if not self.calib_records:
            return 0.0, 0.0
        eps = 1e-10
        # calib_records stores binary (home win, away win) per match.
        # Approximate multinomial log-loss using stored binary predictions.
        model_ll = prior_ll = 0.0
        n = 0
        for market, pred, outcome in self.calib_records:
            prior_p = _PRIOR.get(market, {})
            # Use binary cross-entropy: -(y*log(p) + (1-y)*log(1-p))
            model_ll += outcome * math.log(max(pred, eps)) + (1-outcome) * math.log(max(1-pred, eps))
            # Naive prior binary prediction for this outcome
            if market == "1x2":
                pp = 0.38   # home or away approx
            elif market == "ou25":
                pp = 0.50
            elif market == "btts":
                pp = 0.44
            else:
                pp = 0.50
            prior_ll += outcome * math.log(pp) + (1-outcome) * math.log(1 - pp)
            n += 1
        if n == 0:
            return 0.0, 0.0
        return round(-model_ll / n, 4), round(-prior_ll / n, 4)


# ── Outcome resolvers ─────────────────────────────────────────────────────────

def _resolve_1x2(hg: int, ag: int) -> str:
    if hg > ag: return "home"
    if hg < ag: return "away"
    return "draw"

def _resolve_ou(hg: int, ag: int, line: float) -> str | None:
    tot = hg + ag
    if tot > line: return "over"
    if tot < line: return "under"
    return None  # push — skip

def _resolve_btts(hg: int, ag: int) -> str:
    return "yes" if (hg > 0 and ag > 0) else "no"


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(train_years=("2018",), test_years=("2022",),
                 real_odds: dict | None = None) -> BacktestResult:
    """
    Train on train_years, test on test_years using NAIVE PRIOR as baseline odds.

    real_odds: optional dict keyed by (home_team, away_team) →
               { market_name: {selection: decimal_odds} }
               If provided, real bookmaker odds replace naive prior for that match.
    """
    train_df = load_matches(train_years)
    test_df  = load_matches(test_years)

    model    = DixonColesModel().fit(train_df)
    result   = BacktestResult()
    bankroll = BANKROLL
    result.bankroll_curve.append(bankroll)

    for _, row in test_df.sort_values("date").iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        hg    = int(row["home_goals"])
        ag    = int(row["away_goals"])
        label = f"{home} vs {away}"
        date  = str(row["date"].date())
        real  = real_odds.get((home, away), {}) if real_odds else {}

        # Teams unseen in training → skip (no reliable rating)
        if home not in model.teams or away not in model.teams:
            continue

        # ── 1X2 ──────────────────────────────────────────────────────────────
        mr_probs_raw = match_result(model, home, away)
        mr_probs = _shrink(mr_probs_raw, "1x2")
        mr_odds  = real.get("1x2", _prior_odds("1x2"))
        actual   = _resolve_1x2(hg, ag)

        # Calibration with shrunk probs
        result.calib_records.append(("1x2", mr_probs["home"], int(actual == "home")))
        result.calib_records.append(("1x2", mr_probs["away"], int(actual == "away")))

        for vbet in find_value("1x2", mr_probs, mr_odds, MIN_EDGE):
            stake = min(vbet.half_kelly * bankroll, MAX_STAKE_FRACTION * bankroll)
            won   = (vbet.selection == actual)
            pnl   = stake * (vbet.bk_odds - 1) if won else -stake
            result.bets.append(BetRecord(
                match=label, date=date, market="1x2",
                selection=vbet.selection, model_prob=vbet.model_prob,
                prior_prob=vbet.implied_prob,
                bk_odds=vbet.bk_odds, edge=vbet.edge,
                stake=round(stake, 2), result="win" if won else "loss",
                pnl=round(pnl, 2),
            ))
            bankroll += pnl
            result.bankroll_curve.append(bankroll)

        # ── Over/Under 2.5 ───────────────────────────────────────────────────
        ou_probs_raw = over_under(model, home, away, lines=(2.5,))["2.5"]
        ou_probs  = _shrink(ou_probs_raw, "ou25")
        ou_odds   = real.get("ou25", _prior_odds("ou25"))
        actual_ou = _resolve_ou(hg, ag, 2.5)

        result.calib_records.append(("ou25", ou_probs["over"], int((hg+ag) > 2.5)))

        if actual_ou is not None:
            for vbet in find_value("ou25", ou_probs, ou_odds, MIN_EDGE):
                stake = min(vbet.half_kelly * bankroll, MAX_STAKE_FRACTION * bankroll)
                won   = (vbet.selection == actual_ou)
                pnl   = stake * (vbet.bk_odds - 1) if won else -stake
                result.bets.append(BetRecord(
                    match=label, date=date, market="ou25",
                    selection=vbet.selection, model_prob=vbet.model_prob,
                    prior_prob=vbet.implied_prob,
                    bk_odds=vbet.bk_odds, edge=vbet.edge,
                    stake=round(stake, 2), result="win" if won else "loss",
                    pnl=round(pnl, 2),
                ))
                bankroll += pnl
                result.bankroll_curve.append(bankroll)

        # ── BTTS ─────────────────────────────────────────────────────────────
        bt_probs_raw = btts(model, home, away)
        bt_probs  = _shrink(bt_probs_raw, "btts")
        bt_odds   = real.get("btts", _prior_odds("btts"))
        actual_bt = _resolve_btts(hg, ag)

        result.calib_records.append(("btts", bt_probs["yes"], int(hg > 0 and ag > 0)))

        for vbet in find_value("btts", bt_probs, bt_odds, MIN_EDGE):
            stake = min(vbet.half_kelly * bankroll, MAX_STAKE_FRACTION * bankroll)
            won   = (vbet.selection == actual_bt)
            pnl   = stake * (vbet.bk_odds - 1) if won else -stake
            result.bets.append(BetRecord(
                match=label, date=date, market="btts",
                selection=vbet.selection, model_prob=vbet.model_prob,
                prior_prob=vbet.implied_prob,
                bk_odds=vbet.bk_odds, edge=vbet.edge,
                stake=round(stake, 2), result="win" if won else "loss",
                pnl=round(pnl, 2),
            ))
            bankroll += pnl
            result.bankroll_curve.append(bankroll)

    return result
