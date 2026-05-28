"""
Dominance ELO rating system for international football.

Key design decisions vs standard ELO:

1. Dominance score instead of binary W/D/L
   - sigmoid(0.7 * effective_goal_diff) maps:
       +5 GD  → 0.97  (dominant)
       +3 GD  → 0.88
       +1 GD  → 0.67
        0 GD  → 0.50  (draw)
       -1 GD  → 0.33
   - When xG available, blend: eff = 0.5*goal_diff + 0.5*xg_diff
     (removes finishing luck from the rating update)

2. K-factor scaled by match importance
   WC knockout:   K=64
   WC group:      K=56
   Major final:   K=56
   Qualifier:     K=40
   Nations Lgue:  K=36
   Friendly:      K=16

3. Time decay applied during fit
   Recent form weighted more than old results.
   A team's ELO is a rolling estimate of current strength.

4. Use as prior regularisation in Dixon-Coles
   Teams with sparse data are pulled toward their ELO-implied strength
   rather than the global average (zero). This fixes the "unknown team = average"
   assumption by using ELO as a better prior.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class DominanceELO:

    INITIAL = 1500.0

    _K_MAP = [
        # (substring, K value) — checked in order, first match wins
        ("world cup",   56),   # WC group default; knockout bumped below
        ("qualifier",   40),
        ("euro",        44),
        ("copa",        44),
        ("afcon",       40),
        ("africa cup",  40),
        ("nations league", 36),
        ("confederation", 36),
        ("gold cup",    36),
        ("asian cup",   36),
        ("friendly",    16),
    ]

    def __init__(self):
        self.ratings: dict[str, float] = {}
        self._history: pd.DataFrame | None = None

    # ── Core maths ────────────────────────────────────────────────────────────

    @staticmethod
    def dominance_score(
        hg: float, ag: float,
        hxg: float | None = None,
        axg: float | None = None,
        scale: float = 0.7,
    ) -> float:
        """
        Score ∈ (0, 1) reflecting home team's dominance.
        Blends goal difference with xG difference when available.
        """
        gd = float(hg) - float(ag)
        if hxg is not None and axg is not None and not (np.isnan(hxg) or np.isnan(axg)):
            xgd = float(hxg) - float(axg)
            eff = 0.5 * gd + 0.5 * xgd
        else:
            eff = gd
        return float(1.0 / (1.0 + np.exp(-scale * eff)))

    @staticmethod
    def expected_score(r_home: float, r_away: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))

    def _k_factor(self, stage: str) -> float:
        s = str(stage).lower()
        # Detect WC knockout
        if "world cup" in s and any(
            kw in s for kw in ("final", "semi", "quarter", "round of", "knockout", "r16")
        ):
            return 64.0
        for substr, k in self._K_MAP:
            if substr in s:
                return float(k)
        return 32.0  # default

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        decay: float = 0.0,
        reference_date=None,
    ) -> "DominanceELO":
        """
        Process matches in chronological order to build ratings.

        df must have: date, home_team, away_team, home_goals, away_goals, stage
        Optionally:   home_xg, away_xg

        decay: if > 0, scale K by exp(-decay * days_ago / 365.25) so that
               recent results update the rating more than old ones.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        ref = reference_date or df["date"].max()
        if decay > 0:
            days_ago = (ref - df["date"]).dt.days.to_numpy(dtype=float)
            time_weights = np.exp(-decay * days_ago / 365.25)
        else:
            time_weights = np.ones(len(df))

        rows = []
        for i, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            hg   = float(row["home_goals"])
            ag   = float(row["away_goals"])
            hxg  = float(row["home_xg"]) if "home_xg" in row and not pd.isna(row.get("home_xg")) else None
            axg  = float(row["away_xg"]) if "away_xg" in row and not pd.isna(row.get("away_xg")) else None
            stage = str(row.get("stage", ""))

            r_h = self.ratings.get(home, self.INITIAL)
            r_a = self.ratings.get(away, self.INITIAL)

            E = self.expected_score(r_h, r_a)
            S = self.dominance_score(hg, ag, hxg, axg)
            K = self._k_factor(stage) * float(time_weights[i])

            delta = K * (S - E)
            self.ratings[home] = r_h + delta
            self.ratings[away] = r_a - delta

            rows.append({
                "date":  row["date"],
                "home":  home,
                "away":  away,
                "hg": hg, "ag": ag,
                "hxg": hxg, "axg": axg,
                "K": round(K, 1),
                "E_home": round(E, 3),
                "S_home": round(S, 3),
                "delta":  round(delta, 1),
                "elo_home_after": round(self.ratings[home], 1),
                "elo_away_after": round(self.ratings[away], 1),
            })

        self._history = pd.DataFrame(rows)
        return self

    # ── Outputs ───────────────────────────────────────────────────────────────

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.INITIAL)

    def normalised(self) -> dict[str, float]:
        """Return ratings normalised to zero-mean unit-std (for use as model prior)."""
        vals = np.array(list(self.ratings.values()))
        mu, sd = vals.mean(), vals.std() + 1e-8
        return {t: (v - mu) / sd for t, v in self.ratings.items()}

    def table(self, teams: list[str] | None = None) -> pd.DataFrame:
        """Leaderboard sorted by ELO rating."""
        ts = teams or sorted(self.ratings)
        rows = []
        for rank, t in enumerate(
            sorted(ts, key=lambda x: self.ratings.get(x, self.INITIAL), reverse=True), 1
        ):
            r = self.ratings.get(t, self.INITIAL)
            rows.append({"Rank": rank, "Team": t, "ELO": round(r)})
        return pd.DataFrame(rows)

    def prior_attack(self) -> dict[str, float]:
        """
        Convert ELO to an attack-parameter prior for Dixon-Coles initialisation.
        Maps ELO linearly to attack offset ∈ [−0.5, +0.5].
        """
        nrm = self.normalised()
        # clip to ±2 SD then scale to ±0.5 attack units
        return {t: float(np.clip(v, -2, 2)) * 0.25 for t, v in nrm.items()}

    def history(self) -> pd.DataFrame | None:
        return self._history
