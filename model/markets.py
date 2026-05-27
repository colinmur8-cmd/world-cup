"""
Market probability calculations derived from a Dixon-Coles score matrix.

Supported markets:
  - match_result      : Home / Draw / Away (1X2)
  - double_chance     : 1X / 12 / X2
  - over_under        : O/U 1.5, 2.5, 3.5, 4.5
  - btts              : Both Teams to Score Yes/No
  - asian_handicap    : AH lines in [-2, -1.5, -1, -0.5, 0, +0.5, +1, +1.5, +2]
  - correct_score     : top-N scorelines by probability
"""

from __future__ import annotations
import numpy as np
from model.dixon_coles import DixonColesModel


def _mat(model: DixonColesModel, home: str, away: str,
         max_goals: int = 8) -> np.ndarray:
    return model.score_matrix(home, away, max_goals)


# ── 1X2 ──────────────────────────────────────────────────────────────────────

def match_result(model: DixonColesModel, home: str, away: str) -> dict[str, float]:
    mat = _mat(model, home, away)
    n = mat.shape[0]
    p_home = float(np.sum([mat[i, j] for i in range(n) for j in range(n) if i > j]))
    p_draw = float(np.sum([mat[i, i] for i in range(n)]))
    p_away = float(np.sum([mat[i, j] for i in range(n) for j in range(n) if j > i]))
    total  = p_home + p_draw + p_away
    return {"home": p_home / total, "draw": p_draw / total, "away": p_away / total}


# ── Double chance ─────────────────────────────────────────────────────────────

def double_chance(model: DixonColesModel, home: str, away: str) -> dict[str, float]:
    r = match_result(model, home, away)
    return {
        "1X": r["home"] + r["draw"],
        "12": r["home"] + r["away"],
        "X2": r["draw"] + r["away"],
    }


# ── Over/Under ───────────────────────────────────────────────────────────────

def over_under(model: DixonColesModel, home: str, away: str,
               lines: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5)) -> dict[str, dict[str, float]]:
    mat = _mat(model, home, away, max_goals=10)
    n = mat.shape[0]
    out = {}
    for line in lines:
        p_over = float(sum(
            mat[i, j]
            for i in range(n) for j in range(n)
            if i + j > line
        ))
        out[str(line)] = {"over": p_over, "under": 1.0 - p_over}
    return out


# ── BTTS ─────────────────────────────────────────────────────────────────────

def btts(model: DixonColesModel, home: str, away: str) -> dict[str, float]:
    mat = _mat(model, home, away)
    n = mat.shape[0]
    p_yes = float(sum(
        mat[i, j] for i in range(1, n) for j in range(1, n)
    ))
    return {"yes": p_yes, "no": 1.0 - p_yes}


# ── Asian Handicap ────────────────────────────────────────────────────────────

def asian_handicap(model: DixonColesModel, home: str, away: str,
                   line: float) -> dict[str, float]:
    """
    Asian handicap applied to the HOME team.
    line > 0  → home gives goals (favourite)
    line < 0  → home receives goals (underdog)

    Half-lines (.5) have no push possibility.
    Whole-lines (0, 1, 2, …) include a push on that margin.
    Quarter-lines (.25, .75) are split bets — returned as weighted average.
    """
    mat = _mat(model, home, away, max_goals=10)
    n = mat.shape[0]

    def _single_ah(l: float) -> dict[str, float]:
        """Compute AH probabilities for a pure half- or whole-line."""
        p_home_win = p_away_win = p_push = 0.0
        for i in range(n):
            for j in range(n):
                margin = i - j  # positive = home leads
                adj    = margin + l  # positive = home covers
                if abs(l % 1) < 1e-9:  # whole line → push possible
                    if adj > 0:
                        p_home_win += mat[i, j]
                    elif adj < 0:
                        p_away_win += mat[i, j]
                    else:
                        p_push += mat[i, j]
                else:              # half line → no push
                    if adj > 0:
                        p_home_win += mat[i, j]
                    else:
                        p_away_win += mat[i, j]
        return {"home": p_home_win, "away": p_away_win, "push": p_push}

    # Quarter-lines: split evenly between two adjacent half/whole-lines
    frac = abs(line) % 1
    if abs(frac - 0.25) < 1e-9 or abs(frac - 0.75) < 1e-9:
        l1 = np.floor(line * 2) / 2
        l2 = np.ceil(line * 2) / 2
        r1 = _single_ah(l1)
        r2 = _single_ah(l2)
        return {
            "home": 0.5 * r1["home"] + 0.5 * r2["home"],
            "away": 0.5 * r1["away"] + 0.5 * r2["away"],
            "push": 0.5 * r1["push"] + 0.5 * r2["push"],
        }

    return _single_ah(line)


# ── Correct Score (top scorelines) ───────────────────────────────────────────

def correct_score(model: DixonColesModel, home: str, away: str,
                  top_n: int = 15) -> list[tuple[str, float]]:
    mat = _mat(model, home, away, max_goals=8)
    n = mat.shape[0]
    scores = [
        (f"{i}-{j}", float(mat[i, j]))
        for i in range(n) for j in range(n)
    ]
    return sorted(scores, key=lambda x: -x[1])[:top_n]


# ── Convenience: all standard markets ────────────────────────────────────────

def all_markets(model: DixonColesModel, home: str, away: str) -> dict:
    lam, mu = model.expected_goals(home, away)
    return {
        "expected_goals": {"home": round(lam, 3), "away": round(mu, 3)},
        "match_result":   match_result(model, home, away),
        "double_chance":  double_chance(model, home, away),
        "over_under":     over_under(model, home, away),
        "btts":           btts(model, home, away),
        "asian_handicap": {
            str(l): asian_handicap(model, home, away, l)
            for l in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        },
        "correct_score": correct_score(model, home, away),
    }
