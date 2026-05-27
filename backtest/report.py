"""Terminal report + matplotlib equity curve for a BacktestResult."""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from tabulate import tabulate
from backtest.engine import BacktestResult


MARKET_LABELS = {
    "1x2":  "Match Result (1X2)",
    "ou25": "Over/Under 2.5",
    "btts": "Both Teams to Score",
}


def print_report(result: BacktestResult) -> None:
    s = result.summary()
    if "error" in s:
        print(f"[Backtest] {s['error']}")
        return

    sep = "─" * 68
    print(f"\n{'═'*68}")
    print("  WORLD CUP BETTING MODEL — BACKTEST RESULTS")
    print("  Train: WC 2018  │  Test: WC 2022")
    print("  Baseline: naive WC prior odds (38/27/35 — non-circular)")
    print(f"{'═'*68}")

    # ── Overall stats ─────────────────────────────────────────────────────────
    pnl_sign = "+" if s["total_pnl"] >= 0 else ""
    roi_sign  = "+" if s["roi"] >= 0 else ""
    print(f"\n  Starting bankroll : £{BANKROLL:,.0f}")
    print(f"  Final bankroll    : £{s['final_bankroll']:,.2f}  ({pnl_sign}£{s['total_pnl']:.2f})")
    print(f"  Total bets        : {s['total_bets']}")
    print(f"  Win rate          : {s['win_rate']*100:.1f}%  ({s['total_wins']} wins)")
    print(f"  ROI               : {s['roi']*100:+.2f}%")
    print(f"  Sharpe ratio      : {s['sharpe']:.3f}")
    print(f"  Max drawdown      : {s['max_drawdown']*100:.1f}%")

    # ── Log-loss vs naive prior ───────────────────────────────────────────────
    model_ll, prior_ll = result.log_loss()
    improvement = (prior_ll - model_ll) / prior_ll * 100 if prior_ll > 0 else 0
    print(f"\n  Log-loss (lower = better):")
    print(f"    Dixon-Coles model : {model_ll:.4f}")
    print(f"    Naive prior       : {prior_ll:.4f}")
    print(f"    Improvement       : {improvement:+.1f}%")

    # ── Calibration buckets ───────────────────────────────────────────────────
    buckets = result.calibration_report(n_bins=5)
    if buckets:
        calib_rows = [(f"{b.bin_centre*100:.0f}%",
                       f"{b.predicted*100:.1f}%",
                       f"{b.actual*100:.1f}%",
                       b.count)
                      for b in buckets]
        print(f"\n  Calibration (predicted vs actual win rate by probability band):")
        print(tabulate(calib_rows,
                       headers=["Prob band","Avg predicted","Actual freq","N"],
                       tablefmt="rounded_outline",
                       colalign=("right","right","right","right")))

    # ── Per-market breakdown ─────────────────────────────────────────────────
    bm = s["by_market"].copy()
    bm["market"] = bm["market"].map(lambda m: MARKET_LABELS.get(m, m))
    bm["win_rate"] = bm["wins"] / bm["bets"]
    bm["roi_%"]      = (bm["roi"] * 100).map(lambda x: f"{x:+.2f}%")
    bm["win_rate_%"] = (bm["win_rate"] * 100).map(lambda x: f"{x:.1f}%")
    bm["pnl"]        = bm["pnl"].map(lambda x: f"£{x:+.2f}")
    display = ["market", "bets", "wins", "win_rate_%", "roi_%", "pnl"]
    print(f"\n{sep}")
    print(tabulate(bm[display], headers="keys",
                   tablefmt="rounded_outline", showindex=False))

    # ── Top 12 value bets ────────────────────────────────────────────────────
    bets_df = pd.DataFrame([vars(b) for b in result.bets])
    top = (bets_df.sort_values("edge", ascending=False)
                  .head(12)[["date","match","market","selection",
                              "model_prob","prior_prob","bk_odds",
                              "edge","result","pnl"]])
    top = top.copy()
    top["market"]     = top["market"].map(lambda m: MARKET_LABELS.get(m, m))
    top["model_prob"] = (top["model_prob"]*100).map(lambda x: f"{x:.1f}%")
    top["prior_prob"] = (top["prior_prob"]*100).map(lambda x: f"{x:.1f}%")
    top["edge"]       = (top["edge"]*100).map(lambda x: f"{x:+.1f}pp")
    top["pnl"]        = top["pnl"].map(lambda x: f"£{x:+.2f}")
    print(f"\n  Top value bets by edge (model prob vs naive prior):")
    print(tabulate(top, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()


BANKROLL = 1000.0  # must match engine.py


def save_equity_curve(result: BacktestResult,
                      path: str = "backtest_equity_curve.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        curve  = result.bankroll_curve
        bets_df = pd.DataFrame([vars(b) for b in result.bets])

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 2, 2]})
        fig.suptitle("WC 2026 Betting Model — WC 2022 Backtest (Train: WC 2018)\n"
                     "Baseline: naive WC prior odds (non-circular validation)",
                     fontsize=12, fontweight="bold")

        # ── Equity curve ────────────────────────────────────────────────────
        ax1 = axes[0]
        ax1.plot(range(len(curve)), curve, color="#1565C0", linewidth=2, label="Bankroll")
        ax1.axhline(BANKROLL, color="grey", linewidth=1, linestyle="--", alpha=0.6, label="Start")
        ax1.fill_between(range(len(curve)), curve, BANKROLL,
                         where=[c >= BANKROLL for c in curve], alpha=0.15, color="#4CAF50")
        ax1.fill_between(range(len(curve)), curve, BANKROLL,
                         where=[c < BANKROLL for c in curve], alpha=0.15, color="#F44336")
        ax1.set_ylabel("Bankroll (£)", fontsize=10)
        ax1.set_title("Equity Curve", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ── Per-bet P&L bars ────────────────────────────────────────────────
        ax2 = axes[1]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in bets_df["pnl"]]
        ax2.bar(range(len(bets_df)), bets_df["pnl"], color=colors, width=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("P&L per bet (£)", fontsize=10)
        ax2.set_title("Per-Bet P&L", fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

        # ── Calibration ──────────────────────────────────────────────────────
        ax3 = axes[2]
        buckets = result.calibration_report(n_bins=5)
        if buckets:
            preds   = [b.predicted for b in buckets]
            actuals = [b.actual    for b in buckets]
            ax3.scatter(preds, actuals, s=80, color="#1565C0", zorder=5, label="Bins")
            ax3.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect")
            ax3.set_xlabel("Model predicted probability", fontsize=10)
            ax3.set_ylabel("Actual frequency", fontsize=10)
            ax3.set_title("Calibration Plot", fontsize=10)
            ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "Insufficient calibration data",
                     ha="center", va="center", transform=ax3.transAxes)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Equity curve + calibration chart saved → {path}")
    except Exception as e:
        print(f"  [Chart] Could not save chart: {e}")
