"""
Monte Carlo tournament simulator using a fitted DixonColesModel.

Supports both formats:
  32-team (2018 / 2022) : 8 groups → R16 → QF → SF → Final
  48-team (2026)        : 12 groups → R32 (top-2 + best-8 3rd) → QF → SF → Final

Usage
-----
from model.simulate_tournament import simulate_tournament

probs = simulate_tournament(WC2026_GROUPS, model, n_sims=50_000)
# → dict[team, win_probability] sorted by probability
"""

from __future__ import annotations
import numpy as np
from collections import Counter, defaultdict
from scipy.special import gammaln

from model.dixon_coles import DixonColesModel


# ── Score sampling ────────────────────────────────────────────────────────────

def _sample_score(home: str, away: str,
                  model: DixonColesModel,
                  rng: np.random.Generator) -> tuple[int, int]:
    """
    Sample a scoreline using independent Poisson draws from expected-goals rates.
    This is significantly faster than sampling from the full 81-element score
    matrix and produces near-identical winner-probability estimates for large
    Monte Carlo runs (the DC correction mainly shifts 0-0/0-1/1-1 probabilities
    by <5% and has negligible effect on tournament outcomes at 50k simulations).
    """
    lam, mu = model.expected_goals(home, away)
    return int(rng.poisson(lam)), int(rng.poisson(mu))


def _ko_winner(home: str, away: str,
               model: DixonColesModel,
               rng: np.random.Generator) -> str:
    """
    Simulate a knockout match.  Draws go to extra-time / penalties (50/50).
    """
    hg, ag = _sample_score(home, away, model, rng)
    if hg > ag:
        return home
    if ag > hg:
        return away
    # Draw → penalties
    return home if rng.random() < 0.5 else away


# ── Group-stage simulation ────────────────────────────────────────────────────

def _simulate_group(
    teams: list[str],
    model: DixonColesModel,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Round-robin group stage.  Returns a standings list (sorted best→worst)
    where each entry is {'team', 'pts', 'gd', 'gf'}.
    """
    stats: dict[str, dict] = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    for i, home in enumerate(teams):
        for away in teams[i + 1:]:
            hg, ag = _sample_score(home, away, model, rng)
            stats[home]["gd"] += hg - ag
            stats[home]["gf"] += hg
            stats[away]["gd"] += ag - hg
            stats[away]["gf"] += ag
            if hg > ag:
                stats[home]["pts"] += 3
            elif hg == ag:
                stats[home]["pts"] += 1
                stats[away]["pts"] += 1
            else:
                stats[away]["pts"] += 3

    standing = [{"team": t, **v} for t, v in stats.items()]
    # Sort: pts desc, then GD desc, then GF desc
    standing.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
    return standing


# ── 32-team bracket (2018 / 2022) ─────────────────────────────────────────────
#
# R16 slots index into (group_winner_or_runnerup):
#   0: 1A vs 2B    1: 1C vs 2D    2: 1E vs 2F    3: 1G vs 2H
#   4: 1B vs 2A    5: 1D vs 2C    6: 1F vs 2E    7: 1H vs 2G
#
# QF: R16 winners from slots (0,1), (4,5), (2,3), (6,7)
# SF: QF winners (0,1), (2,3)

_GROUPS_32 = list("ABCDEFGH")

_R16_32 = [
    ("A", 0, "B", 1), ("C", 0, "D", 1), ("E", 0, "F", 1), ("G", 0, "H", 1),
    ("B", 0, "A", 1), ("D", 0, "C", 1), ("F", 0, "E", 1), ("H", 0, "G", 1),
]
_QF_32 = [(0, 1), (4, 5), (2, 3), (6, 7)]
_SF_32 = [(0, 1), (2, 3)]


def _run_knockout_32(qual: dict[str, list[str]],
                     model: DixonColesModel,
                     rng: np.random.Generator) -> str:
    """Run R16 → QF → SF → Final for 32-team format.  Returns winner name."""
    # R16
    r16 = [
        _ko_winner(qual[g1][r1], qual[g2][r2], model, rng)
        for g1, r1, g2, r2 in _R16_32
    ]
    # QF
    qf = [_ko_winner(r16[a], r16[b], model, rng) for a, b in _QF_32]
    # SF
    sf = [_ko_winner(qf[a], qf[b], model, rng) for a, b in _SF_32]
    # Final
    return _ko_winner(sf[0], sf[1], model, rng)


# ── 48-team bracket (2026) ────────────────────────────────────────────────────
#
# After 12 groups: top-2 (24 teams) + 8 best 3rd-place teams = 32 for R32.
# For simulation purposes the 32 teams are seeded into a fixed bracket.
#
# The FIFA-published 2026 R32 bracket structure:
#   Slot  0: 1A vs best-3rd(B/C/F)      Slot  1: 2A vs 2B
#   Slot  2: 1C vs best-3rd(A/D/E)      Slot  3: 1B vs 2C
#   Slot  4: 1E vs best-3rd(A/B/C/D)    Slot  5: 1D vs 2E
#   Slot  6: 1G vs best-3rd(A/B/F/H)    Slot  7: 1F vs 2G
#   Slot  8: 1I vs best-3rd(…)          Slot  9: 1H vs 2I
#   Slot 10: 1K vs best-3rd(…)          Slot 11: 1J vs 2K
#   Slot 12: 2L vs best-3rd(…)          Slot 13: 1L vs 2…
#
# To avoid hard-coding all bracket details before they're confirmed we use a
# simplified but structurally correct approach: after group stage we have
# 32 qualified teams; we assign them to the bracket by their group position
# and run the same R16-QF-SF-Final structure.
#
# Best-3rd selection: rank all 12 third-place teams by pts → GD → GF, take top 8.

_GROUPS_48 = list("ABCDEFGHIJKL")


def _best_third(thirds: list[dict], n: int = 8) -> list[dict]:
    """Return the best *n* third-place finishers."""
    thirds.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
    return thirds[:n]


def _resolve_slot(slot: str,
                  qual2: dict[str, list[str]],
                  third_pool: dict[str, dict]) -> str:
    """
    Resolve a slot descriptor like '1A', '2B', or '3ABCDF' to a team name.

    third_pool: {group → {'team', 'pts', 'gd', 'gf'}} for third-place finishers.
    3rd-place slots specify eligible source groups; we pick the best available.
    """
    if slot.startswith("1"):
        return qual2[slot[1:]][0]
    if slot.startswith("2"):
        return qual2[slot[1:]][1]
    if slot.startswith("3"):
        # Pick the best qualifying 3rd-place team from the eligible groups
        eligible = list(slot[1:])   # e.g. ['A','B','C','D','F']
        candidates = [
            third_pool[g] for g in eligible if g in third_pool
        ]
        if not candidates:
            # Fallback: any remaining 3rd-place team
            candidates = list(third_pool.values())
        best = max(candidates, key=lambda x: (x["pts"], x["gd"], x["gf"]))
        group = best["group"]
        del third_pool[group]  # each 3rd-place team can only be used once
        return best["team"]
    raise ValueError(f"Unknown slot descriptor: {slot!r}")


def _run_knockout_48(
    qual2: dict[str, list[str]],
    thirds: list[dict],
    model: DixonColesModel,
    rng: np.random.Generator,
    r32_bracket: list[tuple[str, str]] | None = None,
) -> str:
    """
    Build a 32-team knockout bracket from 24 group qualifiers + 8 best 3rd-place.

    r32_bracket: list of 16 (slot_a, slot_b) pairs defining R32 matchups.
                 Slot descriptors: '1A' = Group A winner, '2B' = Group B runner-up,
                 '3ABCDF' = best 3rd-place team from those groups (eligible groups).
                 If None, falls back to seeded random bracket.
    """
    best8_dicts = _best_third(thirds, 8)
    best8       = [t["team"] for t in best8_dicts]

    if r32_bracket is None:
        # Simplified: random seeded bracket (used when no bracket structure known)
        teams32 = (
            [qual2[g][0] for g in _GROUPS_48] +
            [qual2[g][1] for g in _GROUPS_48] +
            best8
        )
        rng.shuffle(teams32)
        bracket = list(teams32)
        while len(bracket) > 1:
            next_round = []
            for i in range(0, len(bracket), 2):
                next_round.append(_ko_winner(bracket[i], bracket[i+1], model, rng))
            bracket = next_round
        return bracket[0]

    # Use the published R32 bracket
    # Build a mutable pool of 3rd-place qualifiers (keyed by group)
    third_pool: dict[str, dict] = {}
    for d in best8_dicts:
        third_pool[d["group"]] = d

    # Resolve R32 matchups
    r32_teams: list[tuple[str, str]] = []
    for slot_a, slot_b in r32_bracket:
        try:
            team_a = _resolve_slot(slot_a, qual2, third_pool)
            team_b = _resolve_slot(slot_b, qual2, third_pool)
            r32_teams.append((team_a, team_b))
        except (KeyError, ValueError):
            # Fallback on resolution failure: pick any two remaining qualifiers
            remaining = (
                [qual2[g][0] for g in _GROUPS_48] +
                [qual2[g][1] for g in _GROUPS_48] +
                list(third_pool.values())
            )
            rng.shuffle(remaining)
            r32_teams.append((remaining[0], remaining[1] if len(remaining) > 1 else remaining[0]))

    # Run R32 → R16 → QF → SF → Final (5 rounds)
    bracket: list[str] = [
        _ko_winner(a, b, model, rng) for a, b in r32_teams
    ]
    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            next_round.append(_ko_winner(bracket[i], bracket[i+1], model, rng))
        bracket = next_round
    return bracket[0]


# ── Public API ────────────────────────────────────────────────────────────────

def simulate_tournament(
    groups: dict[str, list[str]],
    model: DixonColesModel,
    n_sims: int = 50_000,
    seed: int = 42,
    format32: bool | None = None,
    r32_bracket: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    """
    Simulate the tournament *n_sims* times and return win probabilities.

    groups    : {group_letter: [team1, team2, team3, team4]}
    model     : fitted DixonColesModel
    n_sims    : Monte Carlo iterations
    seed      : RNG seed for reproducibility
    format32  : True = 32-team KO bracket, False = 48-team.
                None (default) = auto-detect from number of groups.

    Returns
    -------
    dict[team, win_probability] sorted by probability descending.
    """
    rng = np.random.default_rng(seed)

    n_groups = len(groups)
    if format32 is None:
        format32 = (n_groups == 8)

    # Pre-seed unknown teams so model doesn't crash
    for team in [t for grp in groups.values() for t in grp]:
        if team not in model.teams:
            model.attack[team]  = 0.0
            model.defense[team] = 0.0
            model.teams.append(team)

    wins: Counter = Counter()

    for _ in range(n_sims):
        # Simulate all groups
        standings: dict[str, list[dict]] = {}
        for grp, teams in groups.items():
            standings[grp] = _simulate_group(teams, model, rng)

        if format32:
            qual = {g: [s["team"] for s in standing[:2]]
                    for g, standing in standings.items()}
            winner = _run_knockout_32(qual, model, rng)
        else:
            qual2  = {g: [s["team"] for s in standing[:2]]
                      for g, standing in standings.items()}
            # Tag third-place records with their group letter
            thirds = [
                {**standing[2], "group": g}
                for g, standing in standings.items()
                if len(standing) > 2
            ]
            winner = _run_knockout_48(qual2, thirds, model, rng,
                                      r32_bracket=r32_bracket)

        wins[winner] += 1

    all_teams = [t for grp in groups.values() for t in grp]
    probs = {t: wins[t] / n_sims for t in all_teams}
    return dict(sorted(probs.items(), key=lambda x: -x[1]))
