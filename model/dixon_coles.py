"""
Dixon-Coles Poisson model — fully vectorised for fast fitting.

Reference: Dixon & Coles (1997).

Upgrades over baseline:
  - xG blend: when StatsBomb xG columns present, fit on
    xg_weight*xG + (1-xg_weight)*goals (removes finishing luck)
  - ELO prior: optional dict of attack offsets (from DominanceELO.prior_attack())
    used as regularisation target so sparse teams are pulled toward their
    true ELO-implied strength rather than the global average

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
            n: int,
            weights: np.ndarray,
            elo_prior: np.ndarray,
            elo_lambda: float) -> float:
    attack  = params[:n]
    defense = params[n:2*n]
    rho     = params[2*n]

    lam = np.exp(attack[home_idx] + defense[away_idx])
    mu  = np.exp(attack[away_idx] + defense[home_idx])

    # Poisson log-PMF (works for continuous hg/ag via gammaln)
    ll_h = hg * np.log(np.maximum(lam, 1e-10)) - lam - gammaln(hg + 1)
    ll_a = ag * np.log(np.maximum(mu,  1e-10)) - mu  - gammaln(ag + 1)

    # Dixon-Coles low-score correction (only valid for integer-like targets)
    tau = np.ones(len(hg))
    m00 = (hg < 0.5) & (ag < 0.5)
    m01 = (hg < 0.5) & (ag < 1.5) & (ag >= 0.5)
    m10 = (hg < 1.5) & (hg >= 0.5) & (ag < 0.5)
    m11 = (hg < 1.5) & (hg >= 0.5) & (ag < 1.5) & (ag >= 0.5)
    tau[m00] = np.maximum(1.0 - lam[m00] * mu[m00] * rho, 1e-10)
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10]  * rho
    tau[m11] = 1.0 - rho

    nll = -np.sum(weights * (np.log(np.maximum(tau, 1e-10)) + ll_h + ll_a))

    # ELO-prior regularisation on attack parameters
    # Pulls each team's attack toward its ELO-implied prior rather than zero
    if elo_lambda > 0:
        nll += elo_lambda * np.sum((attack - elo_prior) ** 2)

    return nll


# ── Model class ───────────────────────────────────────────────────────────────

class DixonColesModel:

    def __init__(self):
        self.teams:  list[str]        = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.rho:    float             = 0.0
        self._fitted                   = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        decay: float = 0.3,
        reference_date=None,
        xg_weight: float = 0.6,
        elo_prior: dict[str, float] | None = None,
        elo_lambda: float = 0.5,
    ) -> "DixonColesModel":
        """
        Fit the model.

        xg_weight : blend xG into the target when columns are available.
                    0.0 = raw goals only  |  1.0 = xG only  |  0.6 = recommended
        elo_prior : dict team → attack offset from DominanceELO.prior_attack().
                    Used as regularisation target (replaces pulling toward zero).
        elo_lambda: L2 penalty weight on deviation from ELO prior.
        """
        teams   = sorted(set(df["home_team"]) | set(df["away_team"]))
        t2i     = {t: i for i, t in enumerate(teams)}
        n       = len(teams)

        hi = np.array([t2i[t] for t in df["home_team"]])
        ai = np.array([t2i[t] for t in df["away_team"]])

        # ── xG blend ─────────────────────────────────────────────────────────
        has_xg = (
            xg_weight > 0
            and "home_xg" in df.columns
            and "away_xg" in df.columns
            and df["home_xg"].notna().any()
        )
        if has_xg:
            hxg = pd.to_numeric(df["home_xg"], errors="coerce")
            axg = pd.to_numeric(df["away_xg"], errors="coerce")
            hg_raw = df["home_goals"].to_numpy(dtype=float)
            ag_raw = df["away_goals"].to_numpy(dtype=float)
            # Only blend where xG is available; fall back to goals otherwise
            hxg_fill = hxg.fillna(pd.Series(hg_raw)).to_numpy()
            axg_fill = axg.fillna(pd.Series(ag_raw)).to_numpy()
            xg_avail = hxg.notna().to_numpy()
            hg = np.where(xg_avail,
                          xg_weight * hxg_fill + (1 - xg_weight) * hg_raw,
                          hg_raw)
            ag = np.where(xg_avail,
                          xg_weight * axg_fill + (1 - xg_weight) * ag_raw,
                          ag_raw)
        else:
            hg = df["home_goals"].to_numpy(dtype=float)
            ag = df["away_goals"].to_numpy(dtype=float)

        # ── Time-decay weights ────────────────────────────────────────────────
        if reference_date is None:
            reference_date = df["date"].max()
        days_ago = (reference_date - df["date"]).dt.days.to_numpy(dtype=float)
        weights  = np.exp(-decay * days_ago / 365.25)
        weights  = weights / weights.mean()

        # ── ELO prior vector ──────────────────────────────────────────────────
        elo_prior_arr = np.zeros(n)
        if elo_prior:
            for i, t in enumerate(teams):
                elo_prior_arr[i] = elo_prior.get(t, 0.0)

        # ── Initialise from ELO prior ─────────────────────────────────────────
        x0 = np.zeros(2 * n + 1)
        x0[:n]  = elo_prior_arr          # attack init from ELO
        x0[2*n] = -0.1

        bounds = ([(None, None)] * (2 * n)) + [(-0.99, 0.99)]

        res = minimize(
            _neg_ll, x0,
            args=(hi, ai, hg, ag, n, weights, elo_prior_arr, elo_lambda),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7},
        )

        att_raw  = res.x[:n]
        def_raw  = res.x[n:2*n]
        att_mean = att_raw.mean()

        self.teams          = teams
        self.attack         = {t: att_raw[i]  - att_mean for i, t in enumerate(teams)}
        self.defense        = {t: def_raw[i]  + att_mean for i, t in enumerate(teams)}
        self.rho            = float(res.x[2*n])
        self.decay          = decay
        self.reference_date = reference_date
        self._fitted        = True
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

