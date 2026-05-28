"""
Generic Poisson rate model for ancillary match statistics:
corners, yellow cards, shots on target, fouls.

Model per-team rates:
    λ_home = μ * exp(attack[home] + defense[away] + home_adv)
    λ_away = μ * exp(attack[away] + defense[home])

Fitted by minimising weighted negative Poisson log-likelihood
(same time-decay approach as Dixon-Coles).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as _poisson


class PoissonRateModel:
    """
    Lightweight Poisson model for non-goal match statistics.

    Parameters
    ----------
    stat_name : str
        Label used for messages (e.g. "corners", "yellow_cards").
    """

    def __init__(self, stat_name: str = "stat"):
        self.stat_name  = stat_name
        self.attack:  dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.base:     float = 0.0   # log of global mean rate
        self.teams:    list[str] = []
        self.global_mean: float = 1.0

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        home_col: str,
        away_col: str,
        decay: float = 0.3,
        reference_date=None,
        l2: float = 0.05,
    ) -> "PoissonRateModel":
        """
        Fit attack/defense rates from a DataFrame of matches.

        Required columns: date, home_team, away_team, <home_col>, <away_col>
        """
        df = df.dropna(subset=["home_team", "away_team", home_col, away_col]).copy()
        if df.empty:
            return self

        df["date"] = pd.to_datetime(df["date"])
        ref = reference_date or df["date"].max()
        days_ago = (ref - df["date"]).dt.days.to_numpy(dtype=float)
        weights  = np.exp(-decay * days_ago / 365.25)
        weights  = weights / weights.mean()

        all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams   = all_teams
        n_teams      = len(all_teams)
        idx          = {t: i for i, t in enumerate(all_teams)}

        h_idx = np.array([idx[t] for t in df["home_team"]])
        a_idx = np.array([idx[t] for t in df["away_team"]])
        hg    = df[home_col].to_numpy(dtype=float)
        ag    = df[away_col].to_numpy(dtype=float)

        self.global_mean = float(np.average(np.concatenate([hg, ag]), weights=np.concatenate([weights, weights])))

        def _neg_ll(params: np.ndarray) -> float:
            base    = params[0]
            home_adv= params[1]
            att     = params[2 : 2 + n_teams]
            defe    = params[2 + n_teams :]

            lam_h = np.exp(base + att[h_idx] + defe[a_idx] + home_adv)
            lam_a = np.exp(base + att[a_idx] + defe[h_idx])

            ll = weights * (
                hg * np.log(np.maximum(lam_h, 1e-9)) - lam_h +
                ag * np.log(np.maximum(lam_a, 1e-9)) - lam_a
            )

            reg = l2 * (np.sum(att**2) + np.sum(defe**2))
            return -np.sum(ll) + reg

        x0 = np.zeros(2 + 2 * n_teams)
        x0[0] = np.log(max(self.global_mean, 0.1))

        res = minimize(
            _neg_ll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        params = res.x
        self.base     = float(params[0])
        self.home_adv = float(params[1])
        att  = params[2 : 2 + n_teams]
        defe = params[2 + n_teams :]

        # Centre parameters so global mean attack = 0
        att_mean = att.mean()
        att  = att  - att_mean
        self.base += att_mean

        for i, team in enumerate(all_teams):
            self.attack[team]  = float(att[i])
            self.defense[team] = float(defe[i])

        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def _safe_att(self, team: str) -> float:
        return self.attack.get(team, 0.0)

    def _safe_def(self, team: str) -> float:
        return self.defense.get(team, 0.0)

    def predict(self, home: str, away: str) -> tuple[float, float]:
        """Return (λ_home, λ_away) expected rates for the statistic."""
        lam_h = np.exp(self.base + self._safe_att(home) + self._safe_def(away) + self.home_adv)
        lam_a = np.exp(self.base + self._safe_att(away) + self._safe_def(home))
        return float(lam_h), float(lam_a)

    # ── Over/Under markets ────────────────────────────────────────────────────

    def over_under(
        self,
        home: str,
        away: str,
        lines: tuple[float, ...] = (8.5, 9.5, 10.5),
    ) -> dict[str, dict[str, float]]:
        """
        Over/Under probabilities for total (home + away) of this statistic.
        Uses the Poisson sum = Poisson(λ_home + λ_away).
        """
        lam_h, lam_a = self.predict(home, away)
        lam_total    = lam_h + lam_a

        out = {}
        for line in lines:
            k   = int(np.floor(line))
            p_under = float(_poisson.cdf(k, lam_total))
            out[str(line)] = {"over": 1.0 - p_under, "under": p_under}
        return out

    def home_over_under(
        self,
        home: str,
        away: str,
        lines: tuple[float, ...] = (4.5, 5.5),
    ) -> dict[str, dict[str, float]]:
        """Over/Under for home team's statistic only."""
        lam_h, _ = self.predict(home, away)
        out = {}
        for line in lines:
            k = int(np.floor(line))
            p_under = float(_poisson.cdf(k, lam_h))
            out[str(line)] = {"over": 1.0 - p_under, "under": p_under}
        return out

    def away_over_under(
        self,
        home: str,
        away: str,
        lines: tuple[float, ...] = (4.5, 5.5),
    ) -> dict[str, dict[str, float]]:
        """Over/Under for away team's statistic only."""
        _, lam_a = self.predict(home, away)
        out = {}
        for line in lines:
            k = int(np.floor(line))
            p_under = float(_poisson.cdf(k, lam_a))
            out[str(line)] = {"over": 1.0 - p_under, "under": p_under}
        return out

    # ── Ratings table ─────────────────────────────────────────────────────────

    def team_ratings(self) -> pd.DataFrame:
        rows = []
        for t in sorted(self.teams):
            lam_neutral, _ = self.predict(t, t)
            rows.append({
                "Team":    t,
                "Attack":  round(self.attack.get(t, 0.0), 3),
                "Defense": round(self.defense.get(t, 0.0), 3),
                f"Avg {self.stat_name}/game": round(lam_neutral, 2),
            })
        return pd.DataFrame(rows).sort_values(f"Avg {self.stat_name}/game", ascending=False)


# ── Factory helpers ───────────────────────────────────────────────────────────

def _build_stat_df(raw: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    """Extract the minimal columns needed for fitting."""
    return raw[["date", "home_team", "away_team", home_col, away_col]].copy()


def fit_corners_model(stats_df: pd.DataFrame, decay: float = 0.3) -> PoissonRateModel:
    m = PoissonRateModel("corners")
    m.fit(_build_stat_df(stats_df, "home_corners", "away_corners"),
          "home_corners", "away_corners", decay=decay)
    return m


def fit_yellow_cards_model(stats_df: pd.DataFrame, decay: float = 0.3) -> PoissonRateModel:
    m = PoissonRateModel("yellow cards")
    m.fit(_build_stat_df(stats_df, "home_yellow", "away_yellow"),
          "home_yellow", "away_yellow", decay=decay)
    return m


def fit_shots_model(stats_df: pd.DataFrame, decay: float = 0.3) -> PoissonRateModel:
    m = PoissonRateModel("shots on target")
    m.fit(_build_stat_df(stats_df, "home_shots_on_target", "away_shots_on_target"),
          "home_shots_on_target", "away_shots_on_target", decay=decay)
    return m
