"""
Dixon-Coles Poisson model — fully vectorised for fast fitting.

Reference: Dixon & Coles (1997).

Parameters per team: attack_i, defense_i.
Identification constraint: attack params centred to zero-mean post-fit.
All WC matches treated as neutral-venue (no home advantage term).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


# ── Vectorised negative log-likelihood ───────────────────────────────────────

def _neg_ll(params: np.ndarray,
            home_idx: np.ndarray, away_idx: np.ndarray,
            hg: np.ndarray, ag: np.ndarray,
            n: int) -> float:
    attack  = params[:n]
    defense = params[n:2*n]
    rho     = params[2*n]

    lam = np.exp(attack[home_idx] + defense[away_idx])
    mu  = np.exp(attack[away_idx] + defense[home_idx])

    # Poisson log-PMF vectorised
    ll_h = hg * np.log(np.maximum(lam, 1e-10)) - lam - gammaln(hg + 1)
    ll_a = ag * np.log(np.maximum(mu,  1e-10)) - mu  - gammaln(ag + 1)

    # Dixon-Coles low-score correction
    tau = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = np.maximum(1.0 - lam[m00] * mu[m00] * rho, 1e-10)
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10]  * rho
    tau[m11] = 1.0 - rho

    return -np.sum(np.log(np.maximum(tau, 1e-10)) + ll_h + ll_a)


# ── Model class ───────────────────────────────────────────────────────────────

class DixonColesModel:

    def __init__(self):
        self.teams:  list[str]        = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.rho:    float             = 0.0
        self._fitted                   = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        teams   = sorted(set(df["home_team"]) | set(df["away_team"]))
        t2i     = {t: i for i, t in enumerate(teams)}
        n       = len(teams)

        hi  = np.array([t2i[t] for t in df["home_team"]])
        ai  = np.array([t2i[t] for t in df["away_team"]])
        hg  = df["home_goals"].to_numpy(dtype=int)
        ag  = df["away_goals"].to_numpy(dtype=int)

        x0 = np.zeros(2 * n + 1)
        x0[2*n] = -0.1   # rho initial guess

        bounds = ([(None, None)] * (2 * n)) + [(-0.99, 0.99)]

        res = minimize(
            _neg_ll, x0,
            args=(hi, ai, hg, ag, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7},
        )

        att_raw  = res.x[:n]
        def_raw  = res.x[n:2*n]
        att_mean = att_raw.mean()

        self.teams   = teams
        self.attack  = {t: att_raw[i]  - att_mean for i, t in enumerate(teams)}
        self.defense = {t: def_raw[i]  + att_mean for i, t in enumerate(teams)}
        self.rho     = float(res.x[2*n])
        self._fitted = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        self._check()
        lam = np.exp(self.attack.get(home, 0.0) + self.defense.get(away, 0.0))
        mu  = np.exp(self.attack.get(away, 0.0) + self.defense.get(home, 0.0))
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str,
                     max_goals: int = 8) -> np.ndarray:
        self._check()
        lam, mu = self.expected_goals(home, away)
        g = np.arange(max_goals + 1)
        mat = np.outer(
            np.exp(g * np.log(max(lam, 1e-10)) - lam - gammaln(g + 1)),
            np.exp(g * np.log(max(mu,  1e-10)) - mu  - gammaln(g + 1)),
        )
        rho = self.rho
        # Low-score correction in-place
        if max_goals >= 0:
            mat[0, 0] *= max(1 - lam * mu * rho, 1e-10)
        if max_goals >= 1:
            mat[0, 1] *= 1 + lam * rho
            mat[1, 0] *= 1 + mu  * rho
            mat[1, 1] *= 1 - rho
        mat = np.clip(mat, 0, None)
        return mat / mat.sum()

    def team_ratings(self) -> pd.DataFrame:
        rows = [(t, self.attack[t], self.defense[t]) for t in sorted(self.teams)]
        return (pd.DataFrame(rows, columns=["team", "attack", "defense"])
                  .sort_values("attack", ascending=False)
                  .reset_index(drop=True))

    def _check(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

    def __repr__(self):
        status = f"fitted {len(self.teams)} teams ρ={self.rho:.4f}" if self._fitted else "not fitted"
        return f"DixonColesModel({status})"
