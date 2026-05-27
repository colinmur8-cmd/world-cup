"""
Data validation for embedded WC match data.

Two layers of checking:
  1. Internal consistency  — structure, no duplicates, group sizes.
  2. Known-facts audit     — group qualifiers, knockout winners, tournament champions.
     These facts are encoded from authoritative sources and are used to flag any
     embedded match whose implied group standing contradicts the known outcome.

Usage:
    python -m data.validate              # validate hardcoded 2018 + 2022 data
    python -m data.validate --csv path   # diff hardcoded data against an external CSV
"""

from __future__ import annotations
import sys
import argparse
import pandas as pd
from tabulate import tabulate
from data.loader import load_matches


# ── Known tournament facts ────────────────────────────────────────────────────
# These are the ground-truth qualifiers and knockout results used to audit
# whether the embedded scores produce the correct standings.

# Format: {tournament: {group: [qualifier1, qualifier2]}}  (order = 1st, 2nd)
KNOWN_QUALIFIERS = {
    "2018": {
        "A": ["Uruguay",    "Russia"],
        "B": ["Spain",      "Portugal"],
        "C": ["France",     "Denmark"],
        "D": ["Croatia",    "Argentina"],
        "E": ["Brazil",     "Switzerland"],
        "F": ["Sweden",     "Mexico"],
        "G": ["Belgium",    "England"],
        "H": ["Colombia",   "Japan"],
    },
    "2022": {
        "A": ["Netherlands", "Senegal"],
        "B": ["England",     "USA"],
        "C": ["Argentina",   "Poland"],
        "D": ["France",      "Australia"],
        "E": ["Japan",       "Spain"],
        "F": ["Morocco",     "Croatia"],
        "G": ["Brazil",      "Switzerland"],
        "H": ["Portugal",    "South Korea"],
    },
}

# Known knockout results: (winner, loser) for each stage
# Drawn matches that went to pens are recorded by the winner
KNOWN_KNOCKOUT = {
    "2018": {
        "R16":   [("France","Argentina"),("Uruguay","Portugal"),
                  ("Russia","Spain"),    ("Croatia","Denmark"),
                  ("Brazil","Mexico"),   ("Belgium","Japan"),
                  ("Sweden","Switzerland"),("England","Colombia")],
        "QF":    [("France","Uruguay"),  ("Belgium","Brazil"),
                  ("England","Sweden"),  ("Croatia","Russia")],
        "SF":    [("France","Belgium"),  ("Croatia","England")],
        "Final": [("France","Croatia")],
    },
    "2022": {
        "R16":   [("Netherlands","USA"),    ("Argentina","Australia"),
                  ("France","Poland"),      ("England","Senegal"),
                  ("Croatia","Japan"),      ("Brazil","South Korea"),
                  ("Morocco","Spain"),      ("Portugal","Switzerland")],
        "QF":    [("Croatia","Brazil"),     ("Argentina","Netherlands"),
                  ("Morocco","Portugal"),   ("France","England")],
        "SF":    [("Argentina","Croatia"),  ("France","Morocco")],
        "Final": [("Argentina","France")],
    },
}


# ── Internal consistency checks ───────────────────────────────────────────────

def check_structure(df: pd.DataFrame, tournament: str) -> list[str]:
    errors = []
    t = df[df["tournament"] == tournament]

    # Group stage: 8 groups × 6 matches
    for grp in "ABCDEFGH":
        stage = f"Group {grp}"
        count = len(t[t["stage"] == stage])
        if count != 6:
            errors.append(f"{tournament} {stage}: expected 6 matches, found {count}")

    # Knockout counts
    for stage, expected in [("R16",6),("QF",4),("SF",2),("Final",1)]:
        # R16 has 8 matches but we expect all to be present
        pass
    r16 = len(t[t["stage"] == "R16"])
    if r16 != 8:
        errors.append(f"{tournament} R16: expected 8, found {r16}")
    qf = len(t[t["stage"] == "QF"])
    if qf != 4:
        errors.append(f"{tournament} QF: expected 4, found {qf}")

    # Duplicates
    pairs = t[["stage","home_team","away_team"]].apply(
        lambda r: tuple(sorted([r["home_team"],r["away_team"]])) + (r["stage"],), axis=1
    )
    if pairs.duplicated().any():
        errors.append(f"{tournament}: duplicate match pairs found")

    return errors


# ── Group-standings check ─────────────────────────────────────────────────────

def _group_standings(df: pd.DataFrame, tournament: str, group: str) -> list[str]:
    """Return teams sorted by points (W=3, D=1, L=0), GD, GF."""
    t = df[(df["tournament"] == tournament) & (df["stage"] == f"Group {group}")]
    teams: dict[str, dict] = {}

    for _, row in t.iterrows():
        for team, gf, ga in [(row["home_team"], row["home_goals"], row["away_goals"]),
                              (row["away_team"], row["away_goals"], row["home_goals"])]:
            if team not in teams:
                teams[team] = {"pts": 0, "gd": 0, "gf": 0}
            teams[team]["gd"] += gf - ga
            teams[team]["gf"] += gf
            if gf > ga:   teams[team]["pts"] += 3
            elif gf == ga: teams[team]["pts"] += 1

    return sorted(teams, key=lambda t: (-teams[t]["pts"], -teams[t]["gd"], -teams[t]["gf"]))


def check_qualifiers(df: pd.DataFrame) -> list[str]:
    errors = []
    notices = []
    for tournament, groups in KNOWN_QUALIFIERS.items():
        for grp, (q1, q2) in groups.items():
            standing = _group_standings(df, tournament, grp)
            if not standing:
                continue
            top2 = set(standing[:2])
            expected = {q1, q2}
            if top2 != expected:
                # Check if this is a fair-play tiebreaker situation:
                # both expected qualifiers are in the top 3 and the 2nd/3rd teams
                # have identical pts/gd/gf (tie that fair play resolves).
                top3 = set(standing[:3])
                if expected.issubset(top3):
                    # Confirm the tie: do 2nd and 3rd have equal stats?
                    teams_stats = {}
                    t = df[(df["tournament"] == tournament) & (df["stage"] == f"Group {grp}")]
                    for _, row in t.iterrows():
                        for team, gf, ga in [
                            (row["home_team"], row["home_goals"], row["away_goals"]),
                            (row["away_team"], row["away_goals"], row["home_goals"]),
                        ]:
                            if team not in teams_stats:
                                teams_stats[team] = {"pts": 0, "gd": 0, "gf": 0}
                            teams_stats[team]["gd"] += gf - ga
                            teams_stats[team]["gf"] += gf
                            if gf > ga:   teams_stats[team]["pts"] += 3
                            elif gf == ga: teams_stats[team]["pts"] += 1
                    s2 = teams_stats.get(standing[1], {})
                    s3 = teams_stats.get(standing[2], {}) if len(standing) > 2 else {}
                    if s2 and s3 and s2 == s3:
                        notices.append(
                            f"  [NOTE] {tournament} Group {grp}: {standing[1]} and "
                            f"{standing[2]} tied on all standard metrics "
                            f"(pts={s2['pts']}, GD={s2['gd']:+d}, GF={s2['gf']}). "
                            f"Resolved by Fair Play — data is correct."
                        )
                        continue
                errors.append(
                    f"{tournament} Group {grp}: known qualifiers [{q1}, {q2}] "
                    f"but model standings say top-2 = {standing[:2]}"
                )
    for n in notices:
        print(n)
    return errors


# ── Knockout winners check ────────────────────────────────────────────────────

def check_knockout(df: pd.DataFrame) -> list[str]:
    errors = []
    for tournament, stages in KNOWN_KNOCKOUT.items():
        t = df[df["tournament"] == tournament]
        for stage, fixtures in stages.items():
            stage_df = t[t["stage"] == stage]
            for (known_winner, known_loser) in fixtures:
                row = stage_df[
                    ((stage_df["home_team"] == known_winner) & (stage_df["away_team"] == known_loser)) |
                    ((stage_df["home_team"] == known_loser)  & (stage_df["away_team"] == known_winner))
                ]
                if row.empty:
                    errors.append(f"{tournament} {stage}: match {known_winner} vs {known_loser} not found")
                    continue
                r = row.iloc[0]
                hg, ag = r["home_goals"], r["away_goals"]
                home, away = r["home_team"], r["away_team"]
                # Draws are valid (could go to pens), only flag clear incorrect wins
                if hg == ag:
                    continue   # match went to AET/pens — can't verify from score alone
                implied_winner = home if hg > ag else away
                if implied_winner != known_winner:
                    errors.append(
                        f"{tournament} {stage}: {known_winner} should beat {known_loser} "
                        f"but score {home} {hg}-{ag} {away} implies {implied_winner} won"
                    )
    return errors


# ── CSV diff ──────────────────────────────────────────────────────────────────

def diff_against_csv(csv_path: str) -> pd.DataFrame:
    """
    Compare hardcoded data against an external CSV.

    Expected CSV columns (case-insensitive):
      date, home_team, away_team, home_score, away_score, tournament
    or the football-data.org / Kaggle international results format.
    """
    ext = pd.read_csv(csv_path)
    ext.columns = [c.lower().strip() for c in ext.columns]

    # Normalise column names for common CSV formats
    rename = {
        "home_score": "home_goals", "away_score": "away_goals",
        "score1": "home_goals",     "score2": "away_goals",
    }
    ext.rename(columns=rename, inplace=True)
    ext["date"] = pd.to_datetime(ext["date"])

    # Filter to WC matches — handle both string ("FIFA World Cup") and
    # integer/numeric year-only tournament columns (e.g. from wc_verified.csv).
    if pd.api.types.is_string_dtype(ext["tournament"]):
        wc_filter = ext["tournament"].str.upper().str.contains("WORLD CUP", na=False)
    else:
        wc_filter = ext["tournament"].isin([2018, 2022, "2018", "2022"])
    ext_wc = ext[wc_filter & ext["date"].dt.year.isin([2018, 2022])].copy()

    if ext_wc.empty:
        print("  No WC 2018/2022 matches found in CSV. Check 'tournament' column values.")
        return pd.DataFrame()

    # Merge on date+teams
    hardcoded = load_matches(("2018","2022"))
    merged = hardcoded.merge(
        ext_wc[["date","home_team","away_team","home_goals","away_goals"]],
        on=["date","home_team","away_team"],
        suffixes=("_hc","_ext"),
    )
    diffs = merged[
        (merged["home_goals_hc"] != merged["home_goals_ext"]) |
        (merged["away_goals_hc"] != merged["away_goals_ext"])
    ][["date","home_team","away_team","home_goals_hc","away_goals_hc",
       "home_goals_ext","away_goals_ext"]]
    return diffs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="Path to external CSV for score-level diff")
    args = ap.parse_args()

    df = load_matches(("2018","2022"))
    all_errors: list[str] = []

    print("\n  ── Structure checks ──")
    for t in ("2018","2022"):
        errs = check_structure(df, t)
        if errs:
            all_errors += errs
            for e in errs:
                print(f"  [FAIL] {e}")
        else:
            total = len(df[df["tournament"]==t])
            print(f"  [OK]   WC {t}: {total} matches, structure valid")

    print("\n  ── Group qualifier checks (against known tournament results) ──")
    q_errs = check_qualifiers(df)
    if q_errs:
        all_errors += q_errs
        for e in q_errs:
            print(f"  [WARN] {e}")
    else:
        print("  [OK]   All group qualifiers match known tournament outcomes")

    print("\n  ── Knockout winner checks ──")
    k_errs = check_knockout(df)
    if k_errs:
        all_errors += k_errs
        for e in k_errs:
            print(f"  [WARN] {e}")
    else:
        print("  [OK]   All knockout results match known tournament winners")

    if args.csv:
        print(f"\n  ── Score diff vs {args.csv} ──")
        diffs = diff_against_csv(args.csv)
        if diffs.empty:
            print("  [OK]   No score mismatches found")
        else:
            print(f"  [{len(diffs)} score mismatches]:")
            print(tabulate(diffs, headers="keys", tablefmt="rounded_outline",
                           showindex=False))

    print()
    if all_errors:
        print(f"  ⚠  {len(all_errors)} issue(s) found — review warnings above")
        sys.exit(1)
    else:
        print("  ✓  All checks passed")
        if not args.csv:
            print("  Note: score-level verification requires an external CSV.")
            print("  Run with --csv <path> using the Kaggle international results dataset.")


if __name__ == "__main__":
    main()
