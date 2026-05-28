"""
Benter-style multi-feature logistic regression for 1X2 prediction.

Reference: Benter (1994) — "Computer based horse race handicapping and wagering systems"

The model implements:
    ln(P_i / (1 - P_i)) = β_0 + β_1*X_1i + β_2*X_2i + ... + β_n*X_ni

For multinomial 3-class outcome (home win / draw / away win).

Features used:
  dc_log_odds_home/away : Dixon-Coles base probabilities (log-odds vs draw)
  elo_diff              : ELO rating gap, normalised to ±1 range
  elo_expected          : P(home win) per ELO alone (independent signal)
  attack_diff           : DC attack strength differential
  defense_diff          : DC defense differential (positive = home edge)
  xg_diff               : DC expected goals differential (λ_home − λ_away)
  form_attack_diff      : Recent form xG scored differential
  form_defense_diff     : Recent form xG conceded differential

Outcome: class 0 = home win, 1 = draw, 2 = away win.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model.dixon_coles import DixonColesModel
from model.elo import DominanceELO
from model.markets import match_result as dc_match_result


FEATURE_NAMES = [
    "dc_log_odds_home",
    "dc_log_odds_away",
    "elo_diff",
    "elo_expected",
    "attack_diff",
    "defense_diff",
    "xg_diff",
    "form_attack_diff",
    "form_defense_diff",
    "is_neutral",        # 1 for WC / neutral-venue tournaments, 0 for home-venue matches
]


def _compute_form(
    team: str,
    df: pd.DataFrame,
    ref_date,
    n: int = 5,
) -> dict[str, float]:
    """Recent form for team in the last n matches before ref_date."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < ref_date)
    )
    recent = df[mask].sort_values("date").tail(n)

    if recent.empty:
        return {"scored": 1.2, "conceded": 1.2, "xg_scored": 1.2, "xg_conceded": 1.2}

    scored = conceded = xg_scored = xg_conceded = 0.0
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            scored   += float(row["home_goals"])
            conceded += float(row["away_goals"])
            hxg = row.get("home_xg")
            axg = row.get("away_xg")
            if pd.notna(hxg) and pd.notna(axg):
                xg_scored   += float(hxg)
                xg_conceded += float(axg)
            else:
                xg_scored   += float(row["home_goals"])
                xg_conceded += float(row["away_goals"])
        else:
            scored   += float(row["away_goals"])
            conceded += float(row["home_goals"])
            axg = row.get("away_xg")
            hxg = row.get("home_xg")
            if pd.notna(axg) and pd.notna(hxg):
                xg_scored   += float(axg)
                xg_conceded += float(hxg)
            else:
                xg_scored   += float(row["away_goals"])
                xg_conceded += float(row["home_goals"])

    k = len(recent)
    return {
        "scored":      scored    / k,
        "conceded":    conceded  / k,
        "xg_scored":   xg_scored  / k,
        "xg_conceded": xg_conceded / k,
    }


def extract_features(
    home: str,
    away: str,
    dc_model: DixonColesModel,
    elo_model: DominanceELO | None,
    df: pd.DataFrame,
    ref_date=None,
    form_window: int = 5,
    is_neutral: bool = True,
) -> np.ndarray:
    """
    Build a 10-dimensional feature vector for one match.
    All signals are combined by the Benter logistic regression.

    is_neutral: True for World Cup / neutral-venue tournaments (default).
                False for home-venue qualifiers, Nations League etc.
    """
    # DC base probabilities
    lam, mu = dc_model.expected_goals(home, away)
    mr = dc_match_result(dc_model, home, away)
    p_h, p_d, p_a = mr["home"], mr["draw"], mr["away"]
    eps = 1e-7
    dc_logo_home = np.log(max(p_h, eps) / max(p_d, eps))
    dc_logo_away = np.log(max(p_a, eps) / max(p_d, eps))

    # ELO signals
    if elo_model is not None:
        elo_h = elo_model.get(home)
        elo_a = elo_model.get(away)
        elo_diff = (elo_h - elo_a) / 400.0
        elo_exp  = DominanceELO.expected_score(elo_h, elo_a)
    else:
        elo_diff = (lam - mu) * 0.3   # rough proxy from DC xG
        elo_exp  = 0.5

    # DC attack/defense differential
    att_diff = dc_model.attack.get(home, 0.0) - dc_model.attack.get(away, 0.0)
    def_diff = dc_model.defense.get(away, 0.0) - dc_model.defense.get(home, 0.0)

    # Recent form differential
    if ref_date is not None and not df.empty:
        fh = _compute_form(home, df, ref_date, form_window)
        fa = _compute_form(away, df, ref_date, form_window)
        form_att  = fh["xg_scored"]   - fa["xg_scored"]
        form_def  = fa["xg_conceded"] - fh["xg_conceded"]
    else:
        form_att = lam - mu
        form_def = 0.0

    return np.array([
        dc_logo_home,
        dc_logo_away,
        elo_diff,
        elo_exp,
        att_diff,
        def_diff,
        lam - mu,
        form_att,
        form_def,
        float(is_neutral),
    ], dtype=float)


class BenterModel:
    """
    Multinomial logistic regression that combines Dixon-Coles + ELO + recent form.

    Usage:
        benter = BenterModel().fit(df, dc_model, elo_model)
        probs  = benter.predict_proba("Brazil", "France", dc_model, elo_model, df)
        # {"home": 0.41, "draw": 0.28, "away": 0.31}
    """

    def __init__(self):
        self._lr:      LogisticRegression | None = None
        self._scaler:  StandardScaler     | None = None
        self._fitted:  bool = False
        self.n_samples_: int = 0
        self.coef_: np.ndarray | None = None     # (3, 9) — for display

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        dc_model: DixonColesModel,
        elo_model: DominanceELO | None = None,
        form_window: int = 5,
        C: float = 0.5,
    ) -> "BenterModel":
        """
        Train on historical matches.

        C : regularisation (lower = stronger regularisation).
            0.5 works well for 2k–10k samples; increase to 1.0 with more data.
        """
        X_rows, y_rows = [], []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            if home not in dc_model.teams or away not in dc_model.teams:
                continue

            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            label = 0 if hg > ag else (1 if hg == ag else 2)  # 0=H 1=D 2=A

            neutral = bool(row.get("is_neutral", False))
            try:
                feat = extract_features(
                    home, away, dc_model, elo_model, df,
                    ref_date=row["date"], form_window=form_window,
                    is_neutral=neutral,
                )
            except Exception:
                continue

            if not np.all(np.isfinite(feat)):
                continue

            X_rows.append(feat)
            y_rows.append(label)

        n = len(X_rows)
        if n < 100:
            raise ValueError(
                f"Benter fit needs ≥100 samples, got {n}. "
                "Increase the training window or use competitive_only=False."
            )

        X = np.array(X_rows)
        y = np.array(y_rows)

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        self._lr = LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=2000,
        )
        self._lr.fit(Xs, y)

        self.coef_      = self._lr.coef_   # (3, 9)
        self.n_samples_ = n
        self._fitted    = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        home: str,
        away: str,
        dc_model: DixonColesModel,
        elo_model: DominanceELO | None = None,
        df: pd.DataFrame | None = None,
        ref_date=None,
        is_neutral: bool = True,
    ) -> dict[str, float]:
        """
        Return {"home": p, "draw": p, "away": p}.
        Falls back to DC match_result if model is not fitted.

        is_neutral : True for WC / neutral-venue matches (default).
        ref_date   : if None and df is provided, defaults to df["date"].max() + 1 day
                     so that form is computed from the most recent available matches.
        """
        if not self._fitted:
            return dc_match_result(dc_model, home, away)

        df_use = df if df is not None else pd.DataFrame()

        # Auto-set ref_date to just after latest training match so form uses all data
        if ref_date is None and not df_use.empty and "date" in df_use.columns:
            ref_date = df_use["date"].max() + pd.Timedelta(days=1)

        feat = extract_features(
            home, away, dc_model, elo_model, df_use, ref_date,
            is_neutral=is_neutral,
        )
        Xs = self._scaler.transform(feat.reshape(1, -1))
        raw = self._lr.predict_proba(Xs)[0]

        classes = list(self._lr.classes_)
        p_h = raw[classes.index(0)] if 0 in classes else 1 / 3
        p_d = raw[classes.index(1)] if 1 in classes else 1 / 3
        p_a = raw[classes.index(2)] if 2 in classes else 1 / 3

        return {"home": float(p_h), "draw": float(p_d), "away": float(p_a)}

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame of β coefficients for each feature × outcome class.
        Useful for understanding which signals the model weights most.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

        classes = ["home_win", "draw", "away_win"]
        rows = []
        for i, cls in enumerate(classes):
            for j, name in enumerate(FEATURE_NAMES):
                rows.append({
                    "outcome": cls,
                    "feature": name,
                    "β":       round(float(self.coef_[i, j]), 3),
                })
        return pd.DataFrame(rows)

    def blend(
        self,
        benter_probs: dict[str, float],
        dc_probs: dict[str, float],
        weight: float = 0.6,
    ) -> dict[str, float]:
        """
        Ensemble: weight * Benter + (1-weight) * DC.
        Renormalises to sum to 1.
        """
        keys = ("home", "draw", "away")
        blended = {
            k: weight * benter_probs[k] + (1 - weight) * dc_probs[k]
            for k in keys
        }
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()}

    def __repr__(self):
        if self._fitted:
            return f"BenterModel(fitted, n={self.n_samples_}, features={len(FEATURE_NAMES)})"
        return "BenterModel(not fitted)"
